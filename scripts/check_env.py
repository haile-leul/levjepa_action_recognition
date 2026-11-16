"""Check that the Python environment is ready, without downloading the model.

    python scripts/check_env.py

Prints the versions of the main packages and the GPU(s) PyTorch can see.
"""

import importlib
import os
import platform
import sys

W = 16
PACKAGES = ["torch", "transformers", "huggingface_hub", "av", "numpy", "yaml", "tqdm", "pytest"]


def main():
    print(f"{'python':<{W}}{platform.python_version()}  ({sys.executable})")

    missing = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<{W}}{getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"{name:<{W}}MISSING")
            missing.append(name)

    if "torch" not in missing:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"{'gpu ' + str(i):<{W}}{props.name}, {props.total_memory / 2**30:.0f} GB")
            x = torch.randn(1024, 1024, device="cuda")
            print(f"{'cuda matmul':<{W}}ok ({tuple((x @ x).shape)})")
        else:
            print(f"{'gpu':<{W}}none visible (fine on a laptop, not on the GPU machine)")

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"{'HF cache':<{W}}{hf_home}")

    if missing:
        sys.exit(f"\nMissing packages: {', '.join(missing)}. Run: pip install -r requirements.txt")
    print("\nAll good.")


if __name__ == "__main__":
    main()
