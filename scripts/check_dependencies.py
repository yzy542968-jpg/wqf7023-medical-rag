from __future__ import annotations

import importlib.util
import json
from pathlib import Path


DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pillow": "PIL",
    "streamlit": "streamlit",
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "pytest": "pytest",
    "python_docx": "docx",
    "radgraph": "radgraph",
}


def main() -> None:
    status = {
        name: importlib.util.find_spec(module) is not None
        for name, module in DEPENDENCIES.items()
    }
    torch_status = {}
    if status["torch"]:
        import torch

        torch_status = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    output = {
        "dependencies": status,
        "torch": torch_status,
        "ready_for_tests": status["pytest"],
        "ready_for_dashboard": all(
            status[name]
            for name in ["numpy", "pandas", "streamlit", "torch", "transformers"]
        ),
        "ready_for_medcpt": all(
            status[name] for name in ["numpy", "torch", "transformers"]
        ),
        "ready_for_manuscript_build": status["python_docx"],
        "ready_for_radgraph": status["radgraph"],
    }
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/dependency_status.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
