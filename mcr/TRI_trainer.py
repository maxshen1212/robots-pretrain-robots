"""
Trainer for VisualEncoder with Masked Behavior Cloning (BC) loss.

Expects a 10-element batch from BaseBufferEpicH5:
    (b_im, b_lang, b_state, b_actions,
     b_mask_l, b_mask_r,
     b_contact_l, b_contact_r,
     b_obj_l, b_obj_r)

The forward pass uses FiLM-conditioned visual features, then manually
routes through bc_trunk -> bc_policy to produce 32D action predictions.
Loss is masked MSE, split by left/right hand visibility.
"""

import torch
import time

epsilon = 1e-8


class Trainer:
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        m = model.module  # unwrap DataParallel

        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()

        (b_im, b_lang, b_state, b_actions,
         b_mask_l, b_mask_r,
         b_contact_l, b_contact_r,
         b_obj_l, b_obj_r) = batch

        t2 = time.time()

        bs = b_im.shape[0]

        # Forward pass: VisualEncoder with FiLM conditioning
        # b_im: (bs, 3, H, W) in [0, 255] range
        # b_lang: (bs, 768) DistilBERT embeddings
        features = model(b_im, lang_embedding=b_lang)  # (bs, outdim)

        full_loss = 0

        t3 = time.time()

        if m.bc_weight > 0.0:
            # Route features through bc_trunk -> bc_policy
            pred_actions = m.bc_policy(m.bc_trunk(features))  # (bs, 32)

            # Split 32D into left/right: interleaved as [xL, yL, xR, yR] per waypoint
            pred_r = pred_actions.reshape(bs, 8, 4)
            gt_r = b_actions.reshape(bs, 8, 4)

            pred_left = pred_r[:, :, :2].reshape(bs, -1)   # (bs, 16)
            pred_right = pred_r[:, :, 2:].reshape(bs, -1)  # (bs, 16)
            gt_left = gt_r[:, :, :2].reshape(bs, -1)       # (bs, 16)
            gt_right = gt_r[:, :, 2:].reshape(bs, -1)      # (bs, 16)

            # Per-element MSE (reduction='none') -> (bs, 16)
            loss_left = m.bc_loss(pred_left, gt_left.detach())
            loss_right = m.bc_loss(pred_right, gt_right.detach())

            # Apply visibility masks: b_mask_l/r are (bs, 1), broadcast to (bs, 16)
            masked_loss_left = (loss_left * b_mask_l).sum()
            masked_loss_right = (loss_right * b_mask_r).sum()

            num_valid = (b_mask_l.sum() * 16 + b_mask_r.sum() * 16) + epsilon
            bc_loss = (masked_loss_left + masked_loss_right) / num_valid

            metrics["bc_loss"] = bc_loss.item()
            metrics["mask_l_ratio"] = b_mask_l.mean().item()
            metrics["mask_r_ratio"] = b_mask_r.mean().item()
            full_loss += m.bc_weight * bc_loss

        metrics["full_loss"] = full_loss.item() if torch.is_tensor(full_loss) else float(full_loss)

        t4 = time.time()
        if not eval:
            m.encoder_opt.zero_grad()
            full_loss.backward()
            m.encoder_opt.step()

        t5 = time.time()
        st = (f"Load time {t1-t0:.3f}, Batch time {t2-t1:.3f}, "
              f"Forward+Loss time {t4-t3:.3f}, Backprop time {t5-t4:.3f}")
        return metrics, st
