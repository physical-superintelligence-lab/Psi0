"""Console-script wrapper around scripts/deploy/upload_ckpt.py.

Exposed as the `upload_ckpt_hf` command via [project.scripts] so a run under
.runs can be pushed to the Hugging Face model repo from the uv environment:

    upload_ckpt_hf .runs/<subfolder>/<run-dir> [--remote-prefix P | --remote-dir D]
                   [--include-all-ckpts] [--include-wandb]
"""

import os
import sys
from pathlib import Path

# repo_root/src/psi/deploy/upload_ckpt_hf.py -> repo_root
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "deploy" / "upload_ckpt.py"


def main() -> None:
    if not _SCRIPT.is_file():
        sys.exit(f"upload script not found: {_SCRIPT}")
    # Replace the current process so signals (Ctrl-C) and the exit code pass through.
    os.execvp(sys.executable, [sys.executable, str(_SCRIPT), *sys.argv[1:]])


if __name__ == "__main__":
    main()
