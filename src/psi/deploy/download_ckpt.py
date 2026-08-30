"""Console-script wrapper around scripts/deploy/rsync_run_dir.sh.

Exposed as the `download_ckpt` command via [project.scripts] so it can
be run from the uv environment with the same positional args as the shell script:

    download_ckpt [remote-host] [remote-project-dir] [run-dir] [ckpt-step] [local-base]
"""

import os
import sys
from pathlib import Path

# repo_root/src/psi/deploy/download_ckpt.py -> repo_root
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "deploy" / "rsync_run_dir.sh"


def main() -> None:
    if not _SCRIPT.is_file():
        sys.exit(f"rsync script not found: {_SCRIPT}")
    # Replace the current process so signals (Ctrl-C) and the exit code pass through.
    os.execvp("bash", ["bash", str(_SCRIPT), *sys.argv[1:]])


if __name__ == "__main__":
    main()
