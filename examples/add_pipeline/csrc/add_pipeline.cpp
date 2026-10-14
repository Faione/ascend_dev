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
// 正常 NPU 调用的函数链：PyTorch 分派 → AddNpu → launch lambda →
// add_pipeline_kernel → KernelAdd::Init → KernelAdd::Process →
// 每个 tile 的 CopyIn → Compute → CopyOut。Meta 张量走独立的 AddMeta 分支。
namespace ascend_add_pipeline {
namespace PipelineAdd {
// 每个 tile 处理 1024 个 float；输入 X/Y 和输出 Z 各有两个 UB 缓冲槽。
constexpr uint32_t kTile = 1024;
constexpr int32_t kBuffers = 2;

class KernelAdd {
public:
    // __aicore__ 表示在 AI Core 上执行；GM_ADDR 是设备全局内存地址。
    // 每个启动的 block 都创建自己的 KernelAdd 和队列，不与其他核共享 UB。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               int64_t length) {
        // GlobalTensor 为 X/Y/Z 的 GM 地址建立可按元素索引的视图，不复制数据。
        xGm.SetGlobalBuffer((__gm__ float*)x, length);
        yGm.SetGlobalBuffer((__gm__ float*)y, length);
        zGm.SetGlobalBuffer((__gm__ float*)z, length);
        // TPipe 为 TQue 分配片上 UB。三个队列各有 2 个槽，
        // 每槽容量 kTile * sizeof(float) 字节；双缓冲本身不保证阶段重叠。
        pipe.InitBuffer(qx, kBuffers, kTile * sizeof(float));
        pipe.InitBuffer(qy, kBuffers, kTile * sizeof(float));
        pipe.InitBuffer(qz, kBuffers, kTile * sizeof(float));
    }

    __aicore__ inline void Process(int64_t begin, int64_t end) {
        // 一个 block/core 只处理宿主侧分配给它的 [begin, end) 区间。
        // 函数调用次序固定；提交本轮三个阶段后再进入本核的下一个 tile。
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
        // AllocTensor 从输入队列申请一个可写的 UB 槽。
        auto xLocal = qx.AllocTensor<float>();
        auto yLocal = qy.AllocTensor<float>();
        // DataCopyExtParams 描述一次搬运：1 个连续块，长度以字节计。
        // DataCopyPadExtParams 配置非对齐尾部的数据搬运行为。
        AscendC::DataCopyExtParams p{};
        p.blockCount = 1;
        p.blockLen = count * sizeof(float); // 字节数，不是元素数
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], p, pad);
        AscendC::DataCopyPad(yLocal, yGm[offset], p, pad);
        // EnQue 把已填充的槽交给计算阶段；消费者通过 DeQue 取得它。
        qx.EnQue(xLocal);
        qy.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        // DeQue 取得 CopyIn 已入队的 X/Y；计算使用的都是 UB 数据。
        auto xLocal = qx.DeQue<float>();
        auto yLocal = qy.DeQue<float>();
        // 输出队列也先申请一个 UB 槽，供向量 Add 写入。
        auto zLocal = qz.AllocTensor<float>();
        // AscendC::Add 执行逐元素向量加法；最后一个参数是元素数。
        AscendC::Add(zLocal, xLocal, yLocal, count); // 元素数
        // 输出入队供 CopyOut 使用；输入槽用完后归还队列。
        qz.EnQue(zLocal);
        qx.FreeTensor(xLocal);
        qy.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count) {
        // DeQue 取得 Compute 产生的 Z；DataCopyPad 将它从 UB 写回 GM。
        auto zLocal = qz.DeQue<float>();
        AscendC::DataCopyExtParams p{};
        p.blockCount = 1;
        p.blockLen = count * sizeof(float);
        AscendC::DataCopyPad(zGm[offset], zLocal, p);
        // 回写完成后归还输出槽，供后续 tile 复用。
        qz.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    // VECIN/VECOUT 标出向量计算单元的输入和输出队列，深度为 2。
    AscendC::TQue<AscendC::QuePosition::VECIN, kBuffers> qx, qy;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kBuffers> qz;
    AscendC::GlobalTensor<float> xGm, yGm, zGm;
};

__global__ __aicore__ void add_pipeline_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t n, int64_t tilesPerCore) {
    // __global__ 标出设备内核入口；GetBlockIdx 返回当前 block 的编号。
    // 每个 AI Core 获得连续的 tilesPerCore 个 tile；末尾按 n 截断。
    int64_t begin = static_cast<int64_t>(AscendC::GetBlockIdx())
                    * tilesPerCore * kTile;
    if (begin >= n) return;
    int64_t end = begin + tilesPerCore * kTile;
    if (end > n) end = n;
    // 各 block 独立初始化 GM 视图和 UB 队列，再处理自己的区间。
    KernelAdd op;
    op.Init(x, y, z, n);
    op.Process(begin, end);
}

void CheckInputs(const torch::Tensor& x, const torch::Tensor& y) {
    // torch::Tensor 封装形状、dtype、设备和数据指针；TORCH_CHECK 失败会抛异常。
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
    // 仅在 PyTorch 选择 Meta 分派键时运行，不是普通 NPU 调用的必经步骤。
    // empty_like 只创建同形状/dtype 的元数据输出，不访问数据或启动内核。
    CheckInputs(x, y);
    return torch::empty_like(x);
}

torch::Tensor AddNpu(const torch::Tensor& x, const torch::Tensor& y) {
    // PrivateUse1 是 torch_npu 使用的 PyTorch 设备分派类型。
    CheckInputs(x, y);
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "expected NPU tensors");
    // RAII 设备保护：临时选中 x 所在 NPU，函数退出时恢复原设备。
    const c10::OptionalDeviceGuard guard(x.device());
    // empty_like 分配与 x 同形状/dtype/设备的未初始化输出；内核随后写满它。
    auto z = torch::empty_like(x);
    const int64_t n = x.numel();
    // 空向量无需启动 AI Core，返回同形状空输出。
    if (n == 0) return z;

    // 平台管理器查询 AIV 核数与每核 UB 容量，决定是否能运行固定 tile。
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
    // stream 是 NPU 命令队列；选择当前 stream 保持与调用方操作的顺序。
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // data_ptr<float>() 获取设备张量的数据地址，不执行 H2D/D2H 拷贝。
    auto xp = (GM_ADDR)x.data_ptr<float>();
    auto yp = (GM_ADDR)y.data_ptr<float>();
    auto zp = (GM_ADDR)z.data_ptr<float>();
    // 这个 C++ lambda 在宿主侧发射设备内核；<<<blocks, ..., stream>>>
    // 表示启动多少 block、使用哪个 NPU stream。
    auto launch = [=]() -> int {
        add_pipeline_kernel<<<static_cast<uint32_t>(blocks), nullptr, stream>>>(
            xp, yp, zp, n, tilesPerCore);
        return 0;
    };
    // torch_npu 的包装入口执行发射 lambda；通常只是把工作排入 stream。
    // 返回 z 后，读取结果的调用方仍需用同步或后续同 stream 操作等待完成。
    at_npu::native::OpCommand::RunOpApi("AddPipeline", launch);
    return z;
}

TORCH_LIBRARY_FRAGMENT(ascend_add_pipeline, m) {
    // 扩展库加载时向 PyTorch dispatcher 注册算子 schema。
    // Python 入口：torch.ops.ascend_add_pipeline.add_pipeline(x, y)。
    m.def("add_pipeline(Tensor x, Tensor y) -> Tensor");
}
TORCH_LIBRARY_IMPL(ascend_add_pipeline, Meta, m) {
    // Meta 张量只调用 AddMeta；不会进入下面的 NPU 实现。
    m.impl("add_pipeline", AddMeta);
}
TORCH_LIBRARY_IMPL(ascend_add_pipeline, PrivateUse1, m) {
    // NPU 张量调用 AddNpu，再由其发射 AI Core 内核。
    m.impl("add_pipeline", AddNpu);
}
} // namespace PipelineAdd
} // namespace ascend_add_pipeline
