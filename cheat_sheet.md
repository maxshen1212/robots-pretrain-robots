# MCR representation learning: training

```bash
pip install 'huggingface_hub>=0.20,<1.0' timm
cd /data/maxshen/robots-pretrain-robots/mcr

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
```
