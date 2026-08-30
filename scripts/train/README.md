# Training container

The training environment (`.venv-psi`) is baked into a container image, so jobs
start without reading a 9GB venv off BeeGFS. **The image holds dependencies
only** — your source stays in your own checkout, bind-mounted at `/workspace/psi0`
at runtime. One image serves every user.

The base is a plain CUDA image (`nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`),
not NGC PyTorch: NGC's bundled python and torch are shadowed completely by
`.venv-psi`, so the `psi` group installs its own torch 2.7 + `nvidia-*` wheels
and NGC's copies were dead weight. CUDA 12.9 and not 13.x because `pyproject`
pins `torch==2.7.0` and the flash-attn wheel is `cu12torch2.7cxx11abiTRUE-cp311`.

## Build

Two equivalent recipes — the Dockerfile and the enroot script. Keep them in
step if you change one; compose is just a wrapper around the Dockerfile.

| where | how |
|---|---|
| machine with docker | `docker build -f scripts/train/Dockerfile -t psi:train .` (from repo root) |
| machine with compose | `docker compose build psi` (equivalent; see `docker-compose.yml`) |
| cluster node with enroot | `bash scripts/train/build-sqsh.sh` |

On the cluster (no docker, no root needed — `--root` is UID 0 *inside* a user
namespace only):

```bash
ssh h9
cd /path/to/psi0
bash scripts/train/build-sqsh.sh              # ~10 min; PSIX_DRY_RUN=1 to just print paths
sudo cp /mnt/beegfs/scratch/$USER/containers/psix.sqsh /mnt/beegfs/containers/psix.sqsh
```

`build-sqsh.sh` runs on the host; `inner.sh` runs inside the container.
It writes to `/mnt/beegfs/scratch/$USER/containers/psix.sqsh` — installing to
`/mnt/beegfs/containers/` needs an admin, since that directory is root-owned.

To convert a docker-built image instead of building on the cluster:

```bash
ENROOT_TEMP_PATH=/data/enroot-tmp \
ENROOT_SQUASH_OPTIONS='-comp zstd -Xcompression-level 10' \
  enroot import -o psix.sqsh dockerd://psi:train
```

Use zstd, not enroot's `-comp lz4 -noD` default — it roughly halves the result
(16GB vs ~33GB when measured on the older NGC-based image).

## Packages

Dependencies come from `pyproject.toml` + `uv.lock` — change them there, never
inside the image:

```bash
uv add <pkg> --group psi     # or edit pyproject.toml, then: uv lock
```

Then rebuild. A shared image requires a **shared lockfile**: a branch pinning a
different torch or python (e.g. `release` is 3.10, this image is 3.11) needs its
own image.

`flash-attn` is deliberately *not* in `pyproject.toml`. PyPI ships only an sdist
(~1h compile), so both recipes install upstream's prebuilt wheel, pinned to
match the lock (`cu12 / torch2.7 / cxx11abiTRUE / cp311`). Re-pin that URL if the
torch or python pin moves. Because it is absent from `uv.lock`, a later
`uv sync` treats it as extraneous — hence `--inexact` in the Dockerfile.

## Use

`scripts/train/slurm_job.sh` already points at the image:

```
--container-image=/mnt/beegfs/containers/psix.sqsh
--container-mounts=/mnt/beegfs:/mnt/beegfs,"$PROJECT_ROOT":/workspace/psi0
--container-workdir=/workspace/psi0
```

`PROJECT_ROOT` is `$(pwd)` at submit time, so whichever checkout you submit from
is the code that runs. Train scripts pick the venv with:

```bash
source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"
```

which prefers the image venv, falls back to a repo-local `.venv-psi` outside the
container (robot/workstation runs are unaffected), and honours `PSI_VENV`.

## Gotchas

- **GitHub is flaky from the cluster.** `uv` does not retry the `git fetch` for
  the `lerobot` dependency, so the build pre-clones it to
  `/mnt/beegfs/scratch/$USER/mirrors/lerobot.git` and rewrites that one URL with
  `git config insteadOf`. `uv.lock` is untouched; the pinned SHA still resolves.
- **Keep uv's cache node-local.** On BeeGFS, building sdists fails with
  `Device or resource busy` — BeeGFS rejects rename-over-an-open-file. The build
  bind-mounts `/tmp/uv-cache-$USER` at `/uv-cache`, which also keeps the cache
  out of the image.
- **The build needs node-local scratch** for the extracted rootfs — ~60GB was
  enough for the NGC-based image, so it is a safe upper bound for this one.
- **Editable `psi` records an absolute path** (`/workspace/psi0/src/psi`). With a
  second checkout, `import psi` still resolves through the mount, not your cwd.
