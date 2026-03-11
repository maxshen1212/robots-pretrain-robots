# MCR
# TODO: change the parameters
# Refer to train_mcr.sh to launch the training and override hyper-paremeters.

```bash
CUDA_VISIBLE_DEVICES=x,x python train_representation.py hydra/launcher=local \
        hydra/output=local agent.decode_state_weight=1.0 agent.size=50 experiment=<exp_name> \
        doaug=rctraj batch_size=32 datapath=<path_to_dataset> \
        wandbuser=<your_username> wandbproject=<your_proj> use_wandb=false
```
