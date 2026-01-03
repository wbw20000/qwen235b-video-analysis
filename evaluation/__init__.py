# -*- coding: utf-8 -*-
"""评测框架模块"""
from .evaluator import Evaluator, predict_file, EvalResult
from .metrics import compute_metrics

__all__ = ["Evaluator", "predict_file", "EvalResult", "compute_metrics"]
