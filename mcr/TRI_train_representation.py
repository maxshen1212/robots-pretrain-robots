"""
Training loop for VisualEncoder representation learning using EPIC HDF5 data.

Uses BaseBufferEpicH5 dataloader with 10-element tuples and a Masked BC
auxiliary loss computed in TRI_trainer.Trainer.
"""

import warnings

import torchvision

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import hydra
import numpy as np
import torch
from mcr.utils import utils
from mcr.TRI_trainer import Trainer
from mcr.utils.TRI_data_loader import BaseBufferEpicH5
from mcr.utils.logger import Logger
import time

torch.backends.cudnn.benchmark = True


def make_network(cfg):
    model = hydra.utils.instantiate(cfg)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    return model


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        print("Creating Dataloader")
        if self.cfg.dataset == "epic":
            train_iterable = BaseBufferEpicH5(
                dataset_path=self.cfg.datapath,
                num_workers=self.cfg.num_workers,
                split="train",
                datasources=["epic"],
                doaug=self.cfg.doaug,
            )
            val_iterable = BaseBufferEpicH5(
                dataset_path=self.cfg.datapath,
                num_workers=self.cfg.num_workers,
                split="val",
                datasources=["epic"],
                doaug=self.cfg.doaug,
            )
        else:
            raise NameError(f"Invalid Dataset: {self.cfg.dataset}")

        self.train_loader = iter(
            torch.utils.data.DataLoader(
                train_iterable,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
        )
        self.val_loader = iter(
            torch.utils.data.DataLoader(
                val_iterable,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
        )

        ## Init Model
        print("Initializing Model")
        self.model = make_network(cfg.agent).to(self.device)

        self.timer = utils.Timer()
        self._global_step = 0

        ## If reloading existing model
        if cfg.load_snap:
            print("LOADING", cfg.load_snap)
            self.load_snapshot(cfg.load_snap)

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=False, cfg=self.cfg)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step

    def _unpack_batch(self, loader):
        """Unpack 10-element batch from BaseBufferEpicH5 and move to GPU."""
        (
            batch_im, batch_lang, batch_state, batch_actions,
            batch_mask_l, batch_mask_r,
            batch_contact_l, batch_contact_r,
            batch_obj_l, batch_obj_r,
        ) = next(loader)
        return (
            batch_im.to(self.device),
            batch_lang.to(self.device),
            batch_state.to(self.device),
            batch_actions.to(self.device),
            batch_mask_l.to(self.device),
            batch_mask_r.to(self.device),
            batch_contact_l.to(self.device),
            batch_contact_r.to(self.device),
            batch_obj_l.to(self.device),
            batch_obj_r.to(self.device),
        )

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.train_steps, 1)
        eval_freq = self.cfg.eval_freq
        eval_every_step = utils.Every(eval_freq, 1)
        trainer = Trainer(eval_freq)
        # Training Loop
        print("Begin Training")
        while train_until_step(self.global_step):
            # Sample Batch
            t0 = time.time()
            batch = self._unpack_batch(self.train_loader)
            t1 = time.time()

            metrics, st = trainer.update(
                self.model, batch, self.global_step
            )
            t2 = time.time()
            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            if self.global_step % 10 == 0:
                print(self.global_step, metrics)
                print(f"Sample time {t1-t0}, Update time {t2-t1}")
                print(st)

            if eval_every_step(self.global_step):
                with torch.no_grad():
                    batch = self._unpack_batch(self.val_loader)
                    metrics, st = trainer.update(
                        self.model, batch, self.global_step, eval=True
                    )
                    self.logger.log_metrics(
                        metrics, self.global_frame, ty="eval"
                    )
                    print("EVAL", self.global_step, metrics)

                    self.save_snapshot()
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f"snapshot_{self.global_step}.pt"
        global_snapshot = self.work_dir / f"snapshot.pt"
        sdict = {}
        sdict["mcr"] = self.model.state_dict()
        torch.save(sdict, snapshot)
        sdict["global_step"] = self._global_step
        torch.save(sdict, global_snapshot)

    def load_snapshot(self, snapshot_path):
        payload = torch.load(snapshot_path)
        self.model.load_state_dict(payload["mcr"])
        try:
            self._global_step = payload["global_step"]
        except:
            print("No global step found")


@hydra.main(config_path="cfgs", config_name="TRI_config_rep")
def main(cfg):
    from TRI_train_representation import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)

    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot(snapshot)
    workspace.train()


if __name__ == "__main__":
    main()
