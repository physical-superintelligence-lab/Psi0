<h1 align="center">Ψ₀: An Open Foundation Model <br/> Towards Universal Humanoid Loco-Manipulation
</h1>

<p align="center">
  <img src="assets/media/teaser.jpg" alt="Psi0 teaser image" />
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.03233-df2a2a.svg)](https://arxiv.org/abs/2603.12263)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://psi-lab.ai/Psi0)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/songlinwei/psi-model)
[![Data](https://img.shields.io/badge/Hugging%20Face-Data-pink)](https://huggingface.co/datasets/songlinwei/psi-data)
[![License](https://img.shields.io/badge/License-Apache2.0-blue.svg)](./LICENSE)

</div>

Contributors: [Songlin Wei](https://songlin.github.io/), [Hongyi Jing](https://hongyijing.me/), [Boqian Li](https://boqian-li.github.io/), [Zhenyu Zhao](https://zhenyuzhao.com/), [Zhenhao Ni](https://nizhenhao-3.github.io/) , [Sicheng He](https://hesicheng.net/), [Jie Liu](https://jie0530.github.io/), [Xiawei Liu](https://www.xiaweiliu.com/), Kaidi Kang,  Sheng Zang,[Weiduo Yuan](https://weiduoyuan.com/), [Jiageng Mao](https://pointscoder.github.io/), [Marco Pavone](https://profiles.stanford.edu/marco-pavone), Di Huang, [Yue Wang](https://yuewang.xyz/)

-------

$\Psi_0$ is an open vision-language-action (VLA) model for dexterous humanoid loco-manipulation. Our model first learns task semantics and visual representation from large-scale human egocentic videos, and then is post-trained on a smaller amount of real-world teleoperated robot data, to learn general dynamics of the embodiment. 

Our foundation model is capable of acquiring new long-horizontal dexterous loco-manipulation skill by fine-tuning using as few as 80 trajectories. ***Our key finding is that scaling the right data in the right way.***

At the top, the $\Psi_0$ model consists of two end-to-end trained components: a vision–language backbone (System-2) and a multimodal diffusion transformer (System-1) action expert. The backbone is based on Qwen’s Qwen3-VL-2B-Instruct, which extracts vision–language features from observations and instructions. These features condition a flow-based multimodal diffusion transformer inspired by Stable Diffusion 3. The action expert (≈500M parameters) predicts future whole-body action chunks, enabling efficient fusion of visual, linguistic, and action representations. At the lowest level (System-0), an RL-based tracking controller executes the predicted lower-body action commands, ensuring stable and precise physical control.

<p align="center">
  <img src="assets/media/arch.png" alt="Psi0 model" />
</p>

## Table of Contents
<!-- - [Installation](#-environment-setup) -->
<!-- - [Pre- & Post- Training](#-) -->
<!-- - [Data Pre-Processing](#-) -->
- [Finetune Ψ₀ on Unitree G1 Humanoid Robot](#finetune-psi0)
  - [Installation](#installation)
  - [Data Collection](#data-collection)
  - [Fine-Tuning](#training-real)
  - [Open-Loop Evaluation](#open-loop-evaluation)
  - [Deployment](#deployment)
- [Baselines](#baselines)
  - [GR00T N1.6](#groot-n16)
  - [OpenPi π0.5](#openpi-05)
  - [InternVLA-M1](#internvla-m1)
  - [H-RDT](#-)
  - [EgoVLA](#-)
  - [Diffusion Policy](#-)
  - [ACT](#-) 
- [Simulation](#simulation)
  - [Install SIMPLE](#install-simple)
  - [Data Generation](#data-generation)
  - [Fine-Tuning](#training-sim)
  - [Evaluation in SIMPLE](#evaluation-in-simple)
- [Reproduce Ψ₀: Pre-Training and Post-Training](#pre-post-train)
- [Checkpoints](#checkpoints)
- [Troubleshootings](#troubleshootings)
- [Citation](#️-citation)

<a id="finetune-psi0"></a>
## Finetune Ψ₀ on Unitree G1 Humanoid Robot

### Installation

Clone the project and change directory to the project root:
```bash
git clone git@github.com:physical-superintelligence-lab/Psi0.git 
cd Psi0
```
We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python dependencies. Install `uv` if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up the $\Psi_0$ environment:

> ℹ️ We manage the $\Psi_0$ environment and all the baselines through `uv` and they all share the same `src/` code.  See [Environment Management](baselines/README.md) for more details.

```
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match --active
```

Test installation, a version number should be displayed.
```bash
python -c "import psi;print(psi.__version__)"
```

> 🤫 Because the dependent LeRobot version has bug, please apply the patch for each created environment.

Apply the following `LeRobot` patch.
```
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv-psi/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```

### Data Collection
> 📂 We open-sourced all the 9 real-world tasks. You can directly download the data and jump to the [Fine-Tuning](#training-real).

> 🔥 We first release our internal test data collection pipeline which uses Apple Vision Pro to teleoperate Unitree G1 humanoid Robot with two Dex3-1 hands.

See the detailed teleoperation guide here:  
[Real-World Teleoperation Guide](real/README.md)


#### Pre-Processing: Convert Raw Data to LeRobot Format

```
export task=Hug_box_and_move

hf download songlinwei/hfm \
  g1_real_raw/$task.zip \
  --local-dir=$PSI_HOME/data/real_teleop_g1 \
  --repo-type=dataset

unzip $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task.zip -d $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task
```
You should observe similar folder structure:

```
g1_real_raw
└── Hug_box_and_move
    ├── episode_0
    │   ├── color
    │   │   ├── frame_000000.jpg
    │   │   └── ...
    │   └── data.json
    └── ...
```

Edit the task description file with the following format, eg.,
```
vim scripts/data/task_description_dict.json
```
```
{
  "Hug_box_and_move": "Hug box and move."
}
```

Run conversion script
```
python scripts/data/raw_to_lerobot_v2.py \
  --data-root=/hfm/data/real_teleop_g1/g1_real_raw \
  --work-dir=/hfm/data/real \
  --repo-id=psi0-real-g1 \
  --robot-type=g1 \
  --task=$task
```

Calculate stats
```
python scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/real \
  --task=$task
```

Create **$\Psi_0$** format stats (simply a copy for now)
```
cp $PSI_HOME/data/real/$task/meta/stats.json $PSI_HOME/data/real/$task/meta/stats_psi0.json
```

Now it's ready to finetune $\Psi_0$.

> ✈️ If training env is already configured, directly launch training via `scripts/train/psi0/finetune-real-psi0.sh $task`


<a id="training-real"></a>
### Fine-Tuning

> ✔️ Suppose the data is already collected and processed. Now we can proceed to fine-tune the $\Psi_0$ model.

> 📝 Here we illustrate by using the pre-collected data from [Huggingface psi-data](https://huggingface.co/datasets/songlinwei/psi-data/tree/main/real).

Set up the environment variables following `.env.sample`. The environment variables will be loaded by the `dotenv.load_dotenv()` in python.
```
cp .env.sample .env
# and edit the following env variables 
# HF_TOKEN=<YOUR HF READ TOKEN>
# WANDB_API_KEY=<API KEY for wandb logging>
# WANDB_ENTITY=<wandb entity>
# PSI_HOME=<Path where PSI cache/checkpoint/data are located by convention>

source .env
echo $PSI_HOME
```

Download the collected real-world data and extract it:
```
export task=Pick_bottle_and_turn_and_pour_into_cup

hf download songlinwei/psi-data \
  real/$task.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

unzip $PSI_HOME/data/real/$task.zip -d $PSI_HOME/data/real
```
> 👀 If you want to visualize the episode please refer to the [Data Visualization](examples/visualize.md) in the examples.

Launch the training script:
```
scripts/train/psi0/finetune-real-psi0.sh $task
```

> 🖥️ You can always change the GPUs, e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train/...`.  

> ⚠️ Please try to maintain a reasonable global batch size = device batch size x number of GPUs x gradient accumulation step. We use global batch size 128 throughout all the real-world and simulation experiments.


### Open-Loop Evaluation
> Follow the steps in `examples/simple/openloop_eval.ipynb`

Load the training dataset, and run model inference to see how model fits the training data.

### Deployment

#### Serve $\Psi_0$ (RTC mode)

```bash
bash ./scripts/deploy/serve_psi0-rtc.sh
```

#### Start $\Psi_0$ Client (RTC mode)

```bash
bash ./real/scripts/deploy_psi0-rtc.sh
```

For detailed real-world deployment environment setup, please also refer to the dedicated documentation:

[Real-World Teleoperation Guide](real/README.md)


## Baselines

<a id="groot-n16"></a>

### GR00T
Install the env 
```bash
cd src/gr00t; uv sync
```
1. training
```bash
cd src/gr00t
./scripts/train_gr00t.sh --dataset-path /your/lerobot/dataset
```
2. serving a checkpoint
```bash
cd src/gr00t
./scripts/deploy_gr00t.sh
```

3. openloop eval on trained checkpoint using gt
```bash
cd src/gr00t
./scripts/openloop_eval.sh
```


### InternVLA-M1
Install the env 
```bash
cd src/InternVLA-M1; uv sync --python 3.10
```
1. training
```bash
cd src/InternVLA-M1
bash scripts/train_internvla.sh
```
2. serving a checkpoint
```bash
cd src/InternVLA-M1
./scripts/deploy_internvla.sh
```

## Simulation

We use [SIMPLE]() to benchmark $\Psi_0$ and all the baselines.

> 📢 SIMPLE is an easy-to-use humanoid benchmarking simulator built on the MuJoCo physics engine and Isaac Sim rendering.

### Install SIMPLE
[TODO]

### Data Generation
> 📂 We also provide 5 pre-collected whole-body humanoid loco-manipulation tasks at [Huggingface psi-data](https://huggingface.co/datasets/songlinwei/psi-data/tree/main/real). If you want to use the existing simulation data, jump to the [Fine-Tuning](#training-sim)

#### Motion-Planning Based Data Generation
#### Teleoperation in Simulator
[TODO]

<a id="training-sim"></a>
### Fine-Tuning

Download [SIMPLE task data](https://huggingface.co/datasets/songlinwei/psi-data/tree/main/simple) and extract it:

> 💡 Dont forget `source .env` first before following below commands.

```
export task=G1WholebodyBendPick-v0

hf download songlinwei/psi-data \
  simple/$task$.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

unzip $PSI_HOME/data/simple/$task.zip -d $PSI_HOME/data/simple
```

> 👀 If you want to visualize the episode please refer to the [Data Visualization](examples/visualize.md) in the examples.

Start training:

> Please [set up the envrionment variables](#training-real) if not done so yet.

```
bash scripts/train/psi0/finetune-simple-psi0.sh $task
```
The training will create a run dir which is located under `.runs` in the project root.

### Evaluation in SIMPLE

#### Serve $\Psi_0$
```
export run_dir=<the run dir here under folder .runs>
export ckpt_step=<checkpoint step>
uv run --active --group psi --group serve serve_psi0 \
  --host 0.0.0.0 \
  --port 22085 \
  --run-dir=$run_dir \
  --ckpt-step=$ckpt_step
```


Run open-loop evaluation (offline)

[examples/simple/openloop_eval.ipynb](examples/simple/openloop_eval.ipynb)

#### Run the Evaluation Client
> If the server is started on a remote server, run ssh port forward. eg., `ssh -L 22086:localhost:22086 songlin@nebula100`.

> Once port forward is done, open a new terminal to test if server is up `curl -i http://localhost:22085/health`

Launch the eval client through docker
```
GPUs=1 docker compose run eval $task psi0 \
    --host=localhost \
    --port=22085  \
    --sim-mode=mujoco_isaac \
    --headless \
    --max-episode-steps=360 \
    --num-episodes=10 \
    --data-format=lerobot \
    --data-dir=data/$task
```
The policy rollout videos will be found in folder `third_party/SIMPLE/data/evals/psi0`.

> The evaluation for a single episode could take up to 6~10 minutes because SIMPLE use a synchronous rendering API in IsaacSim. See here for [more explanation](#).

<a id="pre-post-train"></a>
## Reproduce Ψ₀: Pre-Training and Post-Training


### Pre-Train VLM

Download and cache the official `Qwen/Qwen3-VL-2B-Instruct` weights.
```
scripts/predownload_qwen3vl.py
```

Pre-train on the [EgoDex dataset](https://github.com/apple/ml-egodex)
```
bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh 
```

Pre-train on [humanoid everyday dataset](https://huggingface.co/datasets/USC-GVL/humanoid-everyday)
```
bash scripts/train/psi0/pretrain-he-psi0-fast.sh
```

Save the pretrained checkpoints once training is done:
```
python scripts/save_pretrain_qwen3vl_backbone.py
```

### Post-Train Action Expert

Download pre-trained `psi-0` VLM backbone
```
python scripts/data/download.py \
  --repo-id=songlinwei/psi-models \
  --remote-dir=pre.fast.egodex.2512241941/pretrained/ckpt_200000 \
  --local-dir=/hfm/cache/checkpoints/psi0/pre.fast.egodex.2512241941.ckpt200k \
  --repo-type=model
```

Post-train on [humanoid everyday (HE) dataset](https://huggingface.co/datasets/USC-GVL/humanoid-everyday)
```
bash scripts/train/psi0/posttrain-he-psi0.sh
```

Save post-trained action header once training is over
```
python scripts/save_posttrain_action_expert.py
```

## Checkpoints

The released checkpoints on [HuggingFace Psi-Model](https://huggingface.co/songlinwei/psi-model) is listed

| Checkpoint | Description | Remote Directory |
|---|---|---|
| $\Psi_0$ VLM<br/>(Baseline) | Pre-trained VLM backbone (EgoDex 200K steps + HE 30K steps) | `psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k` |
| $\Psi_0$ Action Expert<br/>(Baseline) | Post-trained Action Expert On HE | `psi0/postpre.1by1.pad36.2601131206.ckpt.he30k` |

and more variants for ablation studies:
| Checkpoint | Description | Remote Directory |
|---|---|---|
| $\Psi_0$ VLM<br/>(Ablation Study) | Pre-trained VLM backbone only on EgoDex 200K steps | `psi0/pre.fast.egodex.2512241941.ckpt200k` |
| $\Psi_0$ VLM<br/>(Ablation Study) | Pre-trained VLM backbone only on HE 48K steps  | `psi0/pre.abl.only.he.2512311516.48k` |
| $\Psi_0$ VLM<br/>(Ablation Study) | Pre-trained VLM backbone only on 10% EgoDex  | `psi0/pre.abl.ego.10per.2602021632.46k` |
| $\Psi_0$ Action Expert<br/>(Ablation Study) | Post-train on HE by picking pre-trained variant `psi0/pre.abl.only.he.2512311516.48k` | `psi0/postpre.abl.only.he.2602050012` |
| $\Psi_0$ Action Expert<br/>(Ablation Study) | Post-train on HE by picking pre-trained variant `psi0/pre.abl.ego.10per.2602021632.46k` | `psi0/postpre.abl.ego.10per.2602050006` |


Download the selected models

```
hf download songlinwei/psi-models \
  --remote-dir=<remote dictory on huggingface repo>
  --local-dir=$PSI_HOME/cache/checkpoints \
  --repo-type=model
```

## Troubleshootings

1. Lerobot dataset issues: `stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column`

> ⚠️ change `.env/...` to the target env path.

```
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```

2. Fail to install `evdev`, `src/evdev/input.c:10:10: fatal error: Python.h: No such file or directory`

```
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential \
    linux-headers-$(uname -r)
```

3. RuntimeError: Could not load libtorchcodec. Likely causes ...
```
sudo apt-get install ffmpeg
```

4. ImportError: cannot import name 'Deprecated' from 'wandb.proto.wandb_telemetry_pb2' 

re-install `wandb`
```
source .venv-pusht/bin/activate
uv pip uninstall wandb
uv pip install wandb==0.18.0
```

5. support `sm_120` on newer GPUs like `5090` or `RTX 6000`, UserWarning: Ignoring invalid value for boolean flag CUDA_LAUNCH_BLOCKING: truevalid values are 0 or 1.

update `torch` and `flash-attn`
```
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install flash-attn --no-build-isolation
```

6. Failed to download and build `lerobot ... `, Use `git lfs logs last` to view the log.

```
GIT_LFS_SKIP_SMUDGE=1 uv ...
```
## Citation

```
@article{wei2026psi0openfoundationmodel,
  title={$\Psi_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation}, 
  author={Songlin Wei and Hongyi Jing and Boqian Li and Zhenyu Zhao and Jiageng Mao and Zhenhao Ni and Sicheng He and Jie Liu and Xiawei Liu and Kaidi Kang and Sheng Zang and Weiduo Yuan and Marco Pavone and Di Huang and Yue Wang},
  year={2026},
  eprint={2603.12263},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2603.12263}, 
}
```

## License

This project is licensed under the Apache License 2.0.

See the [LICENSE](LICENSE) file for details.