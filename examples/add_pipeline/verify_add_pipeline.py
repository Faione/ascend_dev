import argparse
import torch
import torch_npu
import ascend_add_pipeline


def print_table(rows):
    heading = ("Check", "Result / Details")
    width = max(len(name) for name, _ in [heading, *rows])
    value_width = max(len(value) for _, value in [heading, *rows])
    line = "+-" + "-" * width + "-+-" + "-" * value_width + "-+"
    print(line)
    print(f"| {heading[0]:<{width}} | {heading[1]:<{value_width}} |")
    print(line)
    for name, value in rows:
        print(f"| {name:<{width}} | {value:<{value_width}} |")
    print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    device = f"npu:{args.device}"
    rows = [("torch", torch.__version__),
            ("torch_npu", torch_npu.__version__),
            ("ascend_add_pipeline", ascend_add_pipeline.__file__)]
    stage = "NPU available"

    def expect_error(label, x, y):
        try:
            op(x, y)
        except RuntimeError:
            return
        else:
            raise AssertionError(f"expected rejection: {label}")

    try:
        assert torch.npu.is_available(), "NPU unavailable"
        torch.npu.set_device(device)
        rows.append((stage, f"PASS ({device}: {torch.npu.get_device_name(args.device)})"))
        torch.manual_seed(2026)
        op = torch.ops.ascend_add_pipeline.add_pipeline

        # 空向量、32B 边界、tile 边界、多核多 tile。
        lengths = [0, 1, 7, 8, 9, 31, 32, 33, 1023, 1024, 1025,
                   8192, 65537, 1048576, 1048583]
        for n in lengths:
            stage = f"Add n={n}"
            x_cpu = torch.randn(n, dtype=torch.float32)
            y_cpu = torch.randn(n, dtype=torch.float32)
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            ref = x_cpu + y_cpu
            for _ in range(3):
                out = op(x, y)
                torch.npu.synchronize()
                got = out.cpu()
                assert out.shape == x.shape and out.dtype == x.dtype
                assert out.device == x.device
                torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(x.cpu(), x_cpu, rtol=0, atol=0)
            torch.testing.assert_close(y.cpu(), y_cpu, rtol=0, atol=0)
            max_abs = (got - ref).abs().max().item() if n else 0.0
            rows.append((stage, f"PASS (max_abs_error={max_abs:.3e})"))

        stage = "Add cancellation"
        x = torch.tensor([0., 1., -1., 10., -10., 0.125, -0.125, 2.]).to(device)
        torch.testing.assert_close(op(x, -x).cpu(), torch.zeros(8), rtol=0, atol=0)
        rows.append((stage, "PASS"))

        a = torch.ones(16, device=device)
        invalid = [("shape mismatch", a, a[:8]),
                   ("float16", a.half(), a.half()),
                   ("noncontiguous", a[::2], a[::2]),
                   ("2-D", a.reshape(4, 4), a.reshape(4, 4)),
                   ("requires_grad", a.clone().requires_grad_(), a)]
        for label, left, right in invalid:
            stage = f"Reject {label}"
            expect_error(label, left, right)
            rows.append((stage, "PASS"))
        stage = "NPU synchronize"
        torch.npu.synchronize()
        rows.append(("Overall", "PASS"))
    except Exception as exc:
        message = str(exc).splitlines()
        detail = f"{type(exc).__name__}: {message[0] if message else '(no detail)'}"[:100]
        rows.extend([(stage, f"FAIL ({detail})"), ("Overall", "FAIL")])
        raise
    finally:
        print_table(rows)


if __name__ == "__main__":
    main()
