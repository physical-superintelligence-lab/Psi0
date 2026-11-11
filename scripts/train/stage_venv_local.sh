#!/usr/bin/env bash
# Copy a venv that lives on beegfs to node-local disk, and point its activate at the
# path the container will mount it on. Run once per node by scripts/train/slurm_job.sh:
#
#   stage_venv_local.sh <src venv on beegfs> <dst venv on local disk> <mount path in container>
#
# Why: transformers opens every file under models/ at import time. With 32 ranks doing
# that at once against beegfs, the client intermittently returns ENOENT on a .py file
# that exists (jobs 5470 x2 and 5489 each lost a different node on a different file).
# The psi venv avoids this by living inside psix.sqsh at /workspace/.venv-psi; this
# script gives .venv-dreamzero the same treatment.
set -u
src=$1; dst=$2; mnt=$3
host=$(hostname -s)

mkdir -p "$dst" || { echo "venv stage [$host]: cannot create $dst" >&2; exit 1; }

rc=1
for attempt in 1 2 3; do
    rsync -a --delete "$src/" "$dst/"
    rc=$?
    # 24 = a source file vanished mid-transfer: the very beegfs hiccup this works around.
    [ "$rc" -eq 24 ] || break
    echo "venv stage [$host]: rsync saw a vanished source file (attempt $attempt); retrying" >&2
    sleep 5
done
if [ "$rc" -ne 0 ]; then
    echo "venv stage [$host]: rsync failed with rc=$rc" >&2
    exit "$rc"
fi

# activate hardcodes VIRTUAL_ENV (uv/venv write the absolute beegfs path). Rewrite it to
# the in-container mount path so PATH and ${VIRTUAL_ENV}/lib/... resolve to this copy.
sed -i "s|^VIRTUAL_ENV=.*|VIRTUAL_ENV='$mnt'|" "$dst/bin/activate" || exit 1
grep -q "^VIRTUAL_ENV='$mnt'$" "$dst/bin/activate" || { echo "venv stage [$host]: activate rewrite failed" >&2; exit 1; }

# uv also writes the absolute beegfs interpreter path into every console script's shebang
# (#!/mnt/beegfs/.../.venv-x/bin/python3), so `torchrun` from the local copy would re-exec
# the beegfs python and import from beegfs again (job 5490). Point them at the mount path.
# Only the shebang line is touched, and only in scripts that start with the source path.
n_fixed=0
for f in "$dst"/bin/*; do
    [ -f "$f" ] && [ ! -L "$f" ] || continue
    if head -c 512 "$f" | head -1 | grep -q "^#!$src/bin/python"; then
        sed -i "1s|^#!$src/bin/python|#!$mnt/bin/python|" "$f" || exit 1
        n_fixed=$((n_fixed + 1))
    fi
done
if grep -lq "^#!$src/bin/python" "$dst"/bin/* 2>/dev/null; then
    echo "venv stage [$host]: shebang rewrite incomplete" >&2; exit 1
fi

echo "venv stage [$host]: ok -> $dst ($n_fixed shebangs rewritten, $(du -sh "$dst" 2>/dev/null | cut -f1) used, $(df -h --output=avail "$dst" | tail -1 | tr -d ' ') free on disk)"
