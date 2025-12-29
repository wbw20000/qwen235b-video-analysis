"""
事故分析自动诊断报告系统

用法:
    from reporting import ReportBuilder, RunContext, ClipResult

    # 创建上下文
    context = RunContext()
    context.generate_run_id()
    context.n_clips_cut = 11
    context.n_vlm_analyzed = 0
    context.clip_score_threshold = 0.35

    # 创建clip结果
    clip_results = [
        ClipResult(clip_id="clip-xxx", clip_score=0.12, filter_status="SKIPPED"),
        ...
    ]

    # 生成报告
    builder = ReportBuilder(context, clip_results, output_dir="reports")
    report_dir = builder.build()
"""

from .report_builder import (
    ReportBuilder,
    RunContext,
    ClipResult,
    StageStats,
    RunGate,
    ShortCircuitDiagnostics,
    safe_json_serialize,
    safe_json_dumps,
    safe_json_dump,
    create_report_from_pipeline,
)

from .scorecard import (
    ScorecardBuilder,
    generate_scorecards_from_candidates,
)

__all__ = [
    "ReportBuilder",
    "RunContext",
    "ClipResult",
    "StageStats",
    "RunGate",
    "ShortCircuitDiagnostics",
    "safe_json_serialize",
    "safe_json_dumps",
    "safe_json_dump",
    "create_report_from_pipeline",
    "ScorecardBuilder",
    "generate_scorecards_from_candidates",
]
