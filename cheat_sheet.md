# MCR representation learning: training

```bash
conda activate mcr
pip install 'huggingface_hub>=0.20,<1.0' timm
cd /data/maxshen/robots-pretrain-robots/mcr
wandb login
hf auth login

# overfit on one one episode
CUDA_VISIBLE_DEVICES=2 python TRI_train_representation.py \
    hydra/launcher=local \
    hydra/output=local \
    datapath=/data/maxshen/phantom/data/processed/PutKiwiInCenterOfTable/tri_2d_lang.h5 \
    doaug=resize_vit \
    batch_size=4 \
    train_steps=500 \
    eval_freq=100 \
    lr=1e-4 \
    seed=42 \
    agent.model_name=vit_mae_imagenet_base \
    agent.pretrained=true \
    agent.bc_weight=1.0 \
    agent.use_film_cond=true \
    agent.layernorm_finetune=false \
    experiment=tri_visual_encoder \
    use_wandb=false \
    wandbuser=chihhans-usc \
    wandbproject=SLURM_RPR \
    load_snap=""

# train on all episodes
CUDA_VISIBLE_DEVICES=0,2,4,6 python TRI_train_representation.py \
    hydra/launcher=local \
    hydra/output=local \
    datapath=/data/maxshen/phantom/data/h5output/tri_2d_lang_4tasks_allEpisods.h5 \
    doaug=resize_vit \
    batch_size=128 \
    train_steps=150000 \
    eval_freq=5000 \
    lr=1e-4 \
    seed=42 \
    agent.model_name=vit_mae_imagenet_base \
    agent.pretrained=true \
    agent.bc_weight=1.0 \
    agent.use_film_cond=true \
    agent.layernorm_finetune=false \
    experiment=tri_visual_encoder_all_human_data \
    use_wandb=true \
    wandbuser=chihhans-usc \
    wandbproject=SLURM_RPR \
    load_snap=""

# Author's training script example
python train_representation.py hydra/launcher=local \
hydra/output=local \
dataset=epic_base \
agent.bc_weight=1.0 \
agent.align_state_weight=0.0 \
agent.tcnweight=0.0 \
agent.model_name=resnet18 \
batch_size=xxx \
datapath=<edited EpicKitchens data path> \
experiment=xxx \
wandbuser=xxx​ \
wandbproject=xxx \
use_wandb=true

```
