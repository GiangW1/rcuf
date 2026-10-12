"""Read-only server compatibility report before launching model experiments."""

import importlib.metadata
import json
import subprocess

import torch

from rcuf.engine import environment

report = environment()
report["gpu_count_visible"] = torch.cuda.device_count()
report["gpu_capabilities"] = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
try:
    report["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    report["nvidia_smi"] = None
report["pinned_runtime_ok"] = (
    importlib.metadata.version("transformers") == "5.19.0" and torch.__version__.split("+")[0] == "2.8.0"
)
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    report["fla_import_ok"] = callable(chunk_gated_delta_rule) and callable(fused_recurrent_gated_delta_rule)
except (ImportError, RuntimeError) as error:
    report["fla_import_ok"] = False
    report["fla_error"] = str(error)
print(json.dumps(report, ensure_ascii=False, indent=2))
if not torch.cuda.is_available() or not report["pinned_runtime_ok"] or not report["fla_import_ok"]:
    raise SystemExit("Server check failed; fix environment before model experiments.")
