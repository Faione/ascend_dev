#include <torch/all.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <algorithm>
#include <cstdint>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"
#include "kernel_operator.h"

namespace ascend_add_pipeline {
namespace PipelineAdd {
constexpr uint32_t kTile = 1024;
constexpr int32_t kBuffers = 2;

class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               int64_t length) {
        xGm.SetGlobalBuffer((__gm__ float*)x, length);
        yGm.SetGlobalBuffer((__gm__ float*)y, length);
        zGm.SetGlobalBuffer((__gm__ float*)z, length);
        pipe.InitBuffer(qx, kBuffers, kTile * sizeof(float));
        pipe.InitBuffer(qy, kBuffers, kTile * sizeof(float));
        pipe.InitBuffer(qz, kBuffers, kTile * sizeof(float));
    }

    __aicore__ inline void Process(int64_t begin, int64_t end) {
        for (int64_t offset = begin; offset < end; offset += kTile) {
            uint32_t count = static_cast<uint32_t>(
                end - offset < kTile ? end - offset : kTile);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count) {
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
        auto xLocal = qx.DeQue<float>();
        auto yLocal = qy.DeQue<float>();
        auto zLocal = qz.AllocTensor<float>();
        AscendC::Add(zLocal, xLocal, yLocal, count); // 元素数
        qz.EnQue(zLocal);
        qx.FreeTensor(xLocal);
        qy.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count) {
        auto zLocal = qz.DeQue<float>();
        AscendC::DataCopyExtParams p{};
        p.blockCount = 1;
        p.blockLen = count * sizeof(float);
        AscendC::DataCopyPad(zGm[offset], zLocal, p);
        qz.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kBuffers> qx, qy;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kBuffers> qz;
    AscendC::GlobalTensor<float> xGm, yGm, zGm;
};

__global__ __aicore__ void add_pipeline_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t n, int64_t tilesPerCore) {
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
    if (n == 0) return z;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "cannot query Ascend platform");
    int64_t cores = platform->GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(cores > 0, "no AIV core available");
    TORCH_CHECK(ubBytes >= 3ULL * kBuffers * kTile * sizeof(float),
                "insufficient UB for fixed tile");
    int64_t tiles = (n + kTile - 1) / kTile;
    int64_t blocks = std::min(cores, tiles);
    int64_t tilesPerCore = (tiles + blocks - 1) / blocks;
    // 去除区间起点已经超出数据的多余 block。
    blocks = (tiles + tilesPerCore - 1) / tilesPerCore;
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
