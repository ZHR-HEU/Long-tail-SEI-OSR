# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score,
    precision_recall_fscore_support, top_k_accuracy_score,
    roc_auc_score, average_precision_score
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

class ClassificationAnalyzer:
    def __init__(
        self,
        class_counts: np.ndarray,
        class_names: Optional[List[str]] = None,
        grouping: str = "auto",
        many_thresh: int = 100,
        few_thresh: int = 20,
        q_low: float = 1 / 3,
        q_high: float = 2 / 3
    ):
        self.class_counts = np.asarray(class_counts).astype(int)
        self.num_classes = len(class_counts)
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]
        self.grouping = grouping
        self.many_thresh = many_thresh
        self.few_thresh = few_thresh
        self.q_low = q_low
        self.q_high = q_high
        self._build_groups()

    def _build_groups(self):
        counts = self.class_counts
        if self.grouping == "absolute":
            many = np.where(counts >= self.many_thresh)[0]
            few = np.where(counts <= self.few_thresh)[0]
        elif self.grouping == "quantile":
            lo = np.quantile(counts, self.q_low); hi = np.quantile(counts, self.q_high)
            many = np.where(counts >= hi)[0]; few = np.where(counts <= lo)[0]
        else:
            if counts.max() >= 100:
                many = np.where(counts >= self.many_thresh)[0]
                few = np.where(counts <= self.few_thresh)[0]
            else:
                lo = np.quantile(counts, self.q_low); hi = np.quantile(counts, self.q_high)
                many = np.where(counts >= hi)[0]; few = np.where(counts <= lo)[0]
        medium = np.setdiff1d(np.arange(self.num_classes), np.concatenate([many, few]))
        self.majority_classes, self.medium_classes, self.minority_classes = many, medium, few
        print(f"Class grouping -> majority:{many.tolist()} | medium:{medium.tolist()} | minority:{few.tolist()}")

    def _group_indices(self) -> Dict[str, np.ndarray]:
        return {'majority': self.majority_classes, 'medium': self.medium_classes, 'minority': self.minority_classes}

    def _safe_topk(self, y_true: np.ndarray, prob: np.ndarray, k: int) -> Optional[float]:
        try:
            return float(top_k_accuracy_score(y_true, prob, k=k) * 100.0)
        except Exception:
            return None

    def analyze_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        overall_acc = float((y_true == y_pred).mean() * 100)
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        macro_precision = float(np.mean(precision) * 100)
        macro_recall    = float(np.mean(recall) * 100)
        macro_f1        = float(np.mean(f1) * 100)

        _, _, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        micro_f1 = float(micro_f1 * 100)
        with np.errstate(divide='ignore'):
            gmean = float(np.exp(np.mean(np.log(np.clip(recall, 1e-12, 1.0)))) * 100)

        top5 = macro_auroc = macro_auprc = None
        if prob is not None:
            if prob.ndim == 2 and prob.shape[1] >= 5:
                top5 = self._safe_topk(y_true, prob, k=5)
            if prob.shape[1] <= 200:
                try: macro_auroc = float(roc_auc_score(y_true, prob, multi_class='ovr', average='macro'))
                except Exception: macro_auroc = None
                try:
                    y_true_ovr = np.eye(prob.shape[1])[y_true]
                    ap_list = [average_precision_score(y_true_ovr[:, c], prob[:, c]) for c in range(prob.shape[1])]
                    macro_auprc = float(np.mean(ap_list))
                except Exception:
                    macro_auprc = None

        group_metrics = {}
        groups = self._group_indices()
        for gname, cls_idx in groups.items():
            if len(cls_idx) == 0: group_metrics[gname] = {}; continue
            mask = np.isin(y_true, cls_idx)
            if mask.sum() == 0: group_metrics[gname] = {}; continue
            gp_prec  = float(np.mean(precision[cls_idx]) * 100)
            gp_rec   = float(np.mean(recall[cls_idx]) * 100)
            gp_f1    = float(np.mean(f1[cls_idx]) * 100)
            gp_s     = int(support[cls_idx].sum())
            gp_acc   = float((y_true[mask] == y_pred[mask]).mean() * 100)
            gp_bal   = float(balanced_accuracy_score(y_true[mask], y_pred[mask]) * 100)
            gp_worst = float(np.min(recall[cls_idx]) * 100)
            gp_top5  = None
            if prob is not None and prob.shape[1] >= 5:
                try: gp_top5 = float(top_k_accuracy_score(y_true[mask], prob[mask], k=5) * 100.0)
                except Exception: gp_top5 = None
            group_metrics[gname] = {'accuracy': gp_acc, 'balanced_accuracy': gp_bal, 'precision': gp_prec,
                                    'recall': gp_rec, 'f1': gp_f1, 'support': gp_s,
                                    'worst_class_recall': gp_worst, 'top5': gp_top5}

        cm = confusion_matrix(y_true, y_pred)
        class_metrics = {
            f"{i}": {'precision': float(precision[i] * 100),
                     'recall': float(recall[i] * 100),
                     'f1': float(f1[i] * 100),
                     'support': int(support[i]),
                     'frequency': int(self.class_counts[i])}
            for i in range(self.num_classes)
        }

        return {
            'overall': {
                'accuracy': overall_acc, 'balanced_accuracy': balanced_acc, 'macro_precision': macro_precision,
                'macro_recall': macro_recall, 'macro_f1': macro_f1, 'micro_f1': micro_f1,
                'gmean_recall': gmean, 'top5': top5, 'macro_auroc': macro_auroc, 'macro_auprc': macro_auprc,
            },
            'group_wise': group_metrics,
            'per_class': class_metrics,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': (cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)).tolist(),
            'worst_class_recall': float(np.min(recall) * 100) if recall.size > 0 else 0.0,
            'grouping_meta': {'strategy': self.grouping, 'many_thresh': int(self.many_thresh),
                              'few_thresh': int(self.few_thresh), 'q_low': float(self.q_low), 'q_high': float(self.q_high)}
        }

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str, normalize: bool = False, figsize: Tuple[int, int] = (10, 8)):
        if not HAS_PLOTTING:
            print("Plotting libs unavailable, skip CM plot."); return
        plt.figure(figsize=figsize)
        if normalize:
            cm_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12); fmt='.2f'; title='Normalized Confusion Matrix'
        else:
            cm_plot = cm; fmt='d'; title='Confusion Matrix'
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues')
        plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Confusion matrix saved to: {save_path}")
