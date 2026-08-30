"""Console-script wrapper around scripts/deploy/modelscope_download_run_dir.py.

Exposed as the `download_ckpt_ms` command via [project.scripts] so it can
be run from the uv environment with the same positional args as upload_ckpt_ms,
pulling from a ModelScope repo instead of pushing to one:

    download_ckpt_ms [ms-repo-id] [path-prefix] [run-dir] [ckpt-step] [local-base]
"""

import os
import sys
from pathlib import Path

# repo_root/src/psi/deploy/download_ckpt_ms.py -> repo_root
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "deploy" / "modelscope_download_run_dir.py"


def main() -> None:
    if not _SCRIPT.is_file():
        sys.exit(f"download script not found: {_SCRIPT}")
    # Replace the current process so signals (Ctrl-C) and the exit code pass through.
    os.execvp(sys.executable, [sys.executable, str(_SCRIPT), *sys.argv[1:]])


if __name__ == "__main__":
    main()
