from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(json.dumps({
            "passed": False,
            "error": "PyTorch is not installed in this container",
            "detail": str(exc),
        }, ensure_ascii=False, indent=2))
        return 1

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_vram_gib": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            })

        probe = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float16)
        probe_result = probe @ probe
        finite_probe = bool(torch.isfinite(probe_result).all().item())
    else:
        finite_probe = False

    report = {
        "passed": bool(torch.cuda.is_available() and devices and finite_probe),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16_matrix_probe_finite": finite_probe,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
