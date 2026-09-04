# Finetune Psi-0


```bash
source .venv-psi/bin/activate
set -a; source .env; set +a
```

## Download dataset
```
hf download Psi-X-share/data \
  real/g1_neck30fps_0622.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

unzip /hfm/data/real/g1_neck30fps_0622.zip -d /hfm/data
```

## Launching training
```
bash scripts/train/psi0/finetune-sonic-psi0.sh sonic g1neck30fps622
```
