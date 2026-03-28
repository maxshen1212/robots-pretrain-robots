# MCR
# TODO: change the parameters
# Refer to train_mcr.sh to launch the training and override hyper-paremeters.

```bash
cd /data/maxshen/robots-pretrain-robots/mcr

export WANDB_API_KEY="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"

CUDA_VISIBLE_DEVICES=2 python TRI_train_representation.py \
    hydra/launcher=local \
    hydra/output=local \
    datapath=/data/maxshen/phantom/data/processed/PutKiwiInCenterOfTable/tri_2d_lang.h5 \
    doaug=resize \
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
    wandbproject=masquerade-pretrain \
    load_snap=""
```
