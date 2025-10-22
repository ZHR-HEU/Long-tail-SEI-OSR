# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
import time, numpy as np, torch
import torch.nn as nn
from common import logits_logit_adjustment

def _forward_model(model: nn.Module, x: torch.Tensor):
    out = model(x)
    if isinstance(out, tuple) and len(out) >= 1:
        logits = out[0]; features = out[1] if len(out) > 1 else None
    else:
        logits, features = out, None
    return logits, features

def _compute_loss(criterion: nn.Module, logits: torch.Tensor, target: torch.Tensor, features: Optional[torch.Tensor] = None):
    try:
        return criterion(logits, target, feature=features)
    except TypeError:
        return criterion(logits, target)

def _maybe_apply_eval_posthoc(logits: torch.Tensor, class_counts: np.ndarray, eval_mode: str, tau: float) -> torch.Tensor:
    return logits_logit_adjustment(logits, class_counts, tau=tau) if eval_mode == 'posthoc' else logits

def train_one_epoch(model, loader, criterion, optimizer, device, logger,
                    epoch: int, grad_clip: float = 0.0, use_amp: bool = False, scaler: Optional[torch.cuda.amp.GradScaler] = None):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    scaler = scaler or torch.cuda.amp.GradScaler(enabled=False)
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            amp_enabled = bool(use_amp and device.type == "cuda")
            scaler = scaler or torch.cuda.amp.GradScaler(enabled=amp_enabled)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, features = _forward_model(model, x)
                loss = _compute_loss(criterion, logits, y, features)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            logits, features = _forward_model(model, x)
            loss = _compute_loss(criterion, logits, y, features)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        run_loss += float(loss.item())
        _, pred = logits.max(1)
        total += y.size(0); correct += int(pred.eq(y).sum().item())
        if batch_idx % 50 == 0:
            logger.log_training_step(batch_idx, len(loader), loss.item(), 100.0*correct/max(1,total), optimizer.param_groups[0]['lr'])
    avg_loss = run_loss / max(1, len(loader))
    avg_acc  = 100.0 * correct / max(1, total)
    return {'loss': avg_loss, 'acc': avg_acc}

def evaluate_with_analysis(model, loader, criterion, device, analyzer,
                           class_counts: np.ndarray, eval_logit_adjust: str, eval_logit_tau: float):
    model.eval()
    t0 = time.time()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []
    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits, features = _forward_model(model, x)
            logits_eval = _maybe_apply_eval_posthoc(logits, class_counts, eval_logit_adjust, eval_logit_tau)
            loss = _compute_loss(criterion, logits_eval, y, features)
            val_loss += float(loss.item())
            prob = sm(logits_eval).detach(); _, pred = prob.max(1)
            total += y.size(0); correct += int(pred.eq(y).sum().item())
            all_probs.append(prob.cpu().numpy()); all_preds.extend(pred.cpu().numpy().tolist()); all_targets.extend(y.cpu().numpy().tolist())
    avg_loss = val_loss / max(1, len(loader)); avg_acc  = 100.0 * correct / max(1, total)
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    probs = np.concatenate(all_probs, axis=0) if all_probs else None
    analysis = analyzer.analyze_predictions(np.array(all_targets), np.array(all_preds), prob=probs)
    elapsed = time.time() - t0
    timing = {
        'seconds': float(elapsed),
        'milliseconds': float(elapsed * 1000),
        'throughput_samples_per_sec': float(total / max(1e-9, elapsed))
    }
    return {'loss': avg_loss, 'acc': avg_acc, 'balanced_acc': balanced_acc}, analysis, all_preds, all_targets, timing
