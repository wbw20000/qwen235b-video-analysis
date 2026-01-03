# -*- coding: utf-8 -*-
"""
评测指标计算模块

支持两套评测口径：
- STRICT: 仅 pred_label=="YES" 计为事故；pred_label=="NO" 计为非事故；UNCERTAIN 计为 abstain
- CONSERVATIVE: YES+UNCERTAIN 计为事故
"""
from typing import Dict, List, Any


def compute_metrics(results: List[Any], mode: str = "strict") -> Dict:
    """
    计算评测指标

    Args:
        results: EvalResult列表
        mode: 评测模式
            - "strict": 仅YES为事故，UNCERTAIN作为abstain不计入
            - "conservative": YES+UNCERTAIN都计为事故

    Returns:
        包含各种指标的字典
    """
    tp = 0  # True Positive: 真实事故 & 预测事故
    fp = 0  # False Positive: 真实非事故 & 预测事故 (误报)
    tn = 0  # True Negative: 真实非事故 & 预测非事故
    fn = 0  # False Negative: 真实事故 & 预测非事故 (漏报)
    abstain = 0  # UNCERTAIN的样本数（STRICT模式下不计入）

    # 标签分布统计
    label_dist = {"YES": 0, "NO": 0, "UNCERTAIN": 0}

    for r in results:
        # 获取pred_label（新版）或从predicted推导（兼容旧版）
        pred_label = "NO"
        if hasattr(r, "predict_result") and r.predict_result:
            pred_label = getattr(r.predict_result, "pred_label", "YES" if r.predicted else "NO")
        else:
            pred_label = "YES" if r.predicted else "NO"

        label_dist[pred_label] = label_dist.get(pred_label, 0) + 1

        if mode == "strict":
            # STRICT模式：UNCERTAIN作为abstain
            if pred_label == "UNCERTAIN":
                abstain += 1
                continue  # 不计入TP/FP/TN/FN

            predicted = (pred_label == "YES")
        else:
            # CONSERVATIVE模式：YES+UNCERTAIN都计为事故
            predicted = (pred_label in ("YES", "UNCERTAIN"))

        if r.ground_truth and predicted:
            tp += 1
        elif not r.ground_truth and predicted:
            fp += 1
        elif not r.ground_truth and not predicted:
            tn += 1
        else:  # ground_truth and not predicted
            fn += 1

    # 计算指标
    total = tp + fp + tn + fn
    total_with_abstain = total + abstain

    # Recall (TPR) = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 = 2 * Precision * Recall / (Precision + Recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy = (TP + TN) / Total
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Abstain Rate (STRICT模式下)
    abstain_rate = abstain / total_with_abstain if total_with_abstain > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "abstain": abstain,
        "abstain_rate": abstain_rate,
        "mode": mode,
        "label_dist": label_dist,
    }


def compute_metrics_dual(results: List[Any]) -> Dict:
    """
    同时计算STRICT和CONSERVATIVE两套指标

    Args:
        results: EvalResult列表

    Returns:
        包含两套指标的字典
    """
    strict = compute_metrics(results, mode="strict")
    conservative = compute_metrics(results, mode="conservative")

    return {
        "strict": strict,
        "conservative": conservative,
    }
