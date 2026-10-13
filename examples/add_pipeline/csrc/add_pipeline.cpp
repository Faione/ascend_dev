#include <torch/all.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <algorithm>
#include <cstdint>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"
#include "kernel_operator.h"

// 这个文件同时包含 AI Core 内核与 PyTorch 算子注册。输入限定为一维连续 float32，
// 使教学示例能专注展示 GM → UB → GM 的流水路径。
namespace ascend_add_pipeline {
namespace PipelineAdd {
// 每个 tile 处理 1024 个 float；输入 X/Y 和输出 Z 各有两个 UB 缓冲槽。
constexpr uint32_t kTile = 1024;
constexpr int32_t kBuffers = 2;

class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               int64_t length) {
        // GlobalTensor 只是为全局内存地址建立视图，不复制数据。
        xGm.SetGlobalBuffer((__gm__ float*)x, length);
        yGm.SetGlobalBuffer((__gm__ float*)y, length);
        zGm.SetGlobalBuffer((__gm__ float*)z, length);
        // TPipe 管理队列及 UB；队列深度 2，支持相邻 tile 轮换缓冲槽。
        pipe.InitBuffer(qx, kBuffers, kTile * sizeof(float));
        pipe.InitBuffer(qy, kBuffers, kTile * sizeof(float));
        pipe.InitBuffer(qz, kBuffers, kTile * sizeof(float));
    }

    __aicore__ inline void Process(int64_t begin, int64_t end) {
        // 一个 block/core 只处理宿主侧分配给它的 [begin, end) 区间。
        for (int64_t offset = begin; offset < end; offset += kTile) {
            // 尾 tile 可能不足 1024 个元素，拷贝与计算都使用实际 count。
            uint32_t count = static_cast<uint32_t>(
                end - offset < kTile ? end - offset : kTile);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count) {
        // 先从空闲槽申请 UB，再由 GM 搬入 X/Y，最后入队交给计算阶段。
        auto xLocal = qx.AllocTensor<float>();
        auto yLocal = qy.AllocTensor<float>();
        AscendC::DataCopyExtParams p{};
        p.blockCount = 1;
        p.blockLen = count * sizeof(float); // 字节数，不是元素数
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], p, pad);
        AscendC::DataCopyPad(yLocal, yGm[offset], p, pad);
        qx.EnQue(xLocal);
        qy.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        // 出队等待 CopyIn 完成；输出槽在计算前申请。
        auto xLocal = qx.DeQue<float>();
        auto yLocal = qy.DeQue<float>();
        auto zLocal = qz.AllocTensor<float>();
        AscendC::Add(zLocal, xLocal, yLocal, count); // 元素数
        // 输出入队后，输入槽可释放，供后续 tile 复用。
        qz.EnQue(zLocal);
        qx.FreeTensor(xLocal);
        qy.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count) {
        // 等待计算产物出队，再将 UB 中的 Z 拷回全局内存。
        auto zLocal = qz.DeQue<float>();
        AscendC::DataCopyExtParams p{};
        p.blockCount = 1;
        p.blockLen = count * sizeof(float);
        AscendC::DataCopyPad(zGm[offset], zLocal, p);
        qz.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    // 队列分别代表两个输入流和一个输出流，深度为 2。
    AscendC::TQue<AscendC::QuePosition::VECIN, kBuffers> qx, qy;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kBuffers> qz;
    AscendC::GlobalTensor<float> xGm, yGm, zGm;
};

__global__ __aicore__ void add_pipeline_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t n, int64_t tilesPerCore) {
    // 每个 AI Core 获得连续的 tilesPerCore 个 tile；末尾按 n 截断。
    int64_t begin = static_cast<int64_t>(AscendC::GetBlockIdx())
                    * tilesPerCore * kTile;
    if (begin >= n) return;
    int64_t end = begin + tilesPerCore * kTile;
    if (end > n) end = n;
    KernelAdd op;
    op.Init(x, y, z, n);
    op.Process(begin, end);
}

void CheckInputs(const torch::Tensor& x, const torch::Tensor& y) {
    // 检查放在宿主侧，避免内核处理形状、步长或梯度等额外语义。
    TORCH_CHECK(x.dim() == 1 && y.dim() == 1, "expected 1-D vectors");
    TORCH_CHECK(x.sizes() == y.sizes(), "shape mismatch");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                y.scalar_type() == torch::kFloat32, "expected float32");
    TORCH_CHECK(x.device() == y.device(), "device mismatch");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "expected contiguous inputs");
    TORCH_CHECK(!x.requires_grad() && !y.requires_grad(),
                "autograd is not supported");
}

torch::Tensor AddMeta(const torch::Tensor& x, const torch::Tensor& y) {
    // Meta 实现只推断输出属性，不访问数据，也不启动 NPU 内核。
    CheckInputs(x, y);
    return torch::empty_like(x);
}

torch::Tensor AddNpu(const torch::Tensor& x, const torch::Tensor& y) {
    CheckInputs(x, y);
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "expected NPU tensors");
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = torch::empty_like(x);
    const int64_t n = x.numel();
    // 空向量无需启动 AI Core，返回同形状空输出。
    if (n == 0) return z;

    // 启动前查询设备资源：固定 tile 的三个双缓冲队列必须装入 UB。
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "cannot query Ascend platform");
    int64_t cores = platform->GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(cores > 0, "no AIV core available");
    TORCH_CHECK(ubBytes >= 3ULL * kBuffers * kTile * sizeof(float),
                "insufficient UB for fixed tile");
    // 将总 tile 数尽量均分到 AIV 核；每核仍处理连续区间。
    int64_t tiles = (n + kTile - 1) / kTile;
    int64_t blocks = std::min(cores, tiles);
    int64_t tilesPerCore = (tiles + blocks - 1) / blocks;
    // 去除区间起点已经超出数据的多余 block。
    blocks = (tiles + tilesPerCore - 1) / tilesPerCore;
    // 使用 PyTorch 当前 NPU stream，保持与调用方操作的执行顺序一致。
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)x.data_ptr<float>();
    auto yp = (GM_ADDR)y.data_ptr<float>();
    auto zp = (GM_ADDR)z.data_ptr<float>();
    auto launch = [=]() -> int {
        add_pipeline_kernel<<<static_cast<uint32_t>(blocks), nullptr, stream>>>(
            xp, yp, zp, n, tilesPerCore);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("AddPipeline", launch);
    return z;
}

TORCH_LIBRARY_FRAGMENT(ascend_add_pipeline, m) {
    // Python 入口：torch.ops.ascend_add_pipeline.add_pipeline(x, y)。
    m.def("add_pipeline(Tensor x, Tensor y) -> Tensor");
}
TORCH_LIBRARY_IMPL(ascend_add_pipeline, Meta, m) {
    m.impl("add_pipeline", AddMeta);
}
TORCH_LIBRARY_IMPL(ascend_add_pipeline, PrivateUse1, m) {
    m.impl("add_pipeline", AddNpu);
}
} // namespace PipelineAdd
} // namespace ascend_add_pipeline
