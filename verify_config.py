#!/usr/bin/env python
"""验证TrafficVLM配置是否已更新"""

import sys
sys.path.insert(0, '.')

from traffic_vlm.config import TrafficVLMConfig

print("=" * 80)
print("TrafficVLM 当前配置验证")
print("=" * 80)
print()

config = TrafficVLMConfig()

print("StreamConfig:")
print(f"  motion_min_fg_ratio: {config.stream.motion_min_fg_ratio} (优化前: 0.02)")
print(f"  motion_debounce_frames: {config.stream.motion_debounce_frames} (优化前: 3)")
print(f"  always_sample_interval_seconds: {config.stream.always_sample_interval_seconds}秒")
print()

print("EmbeddingConfig:")
print(f"  top_m_per_template: {config.embedding.top_m_per_template} (优化前: 50)")
print(f"  clip_embedding_frames: {config.embedding.clip_embedding_frames} (优化前: 8)")
print()

print("ClusterConfig:")
print(f"  pre_padding: {config.cluster.pre_padding}秒 (优化前: 3.0)")
print()

print("DetectorConfig:")
print(f"  enabled: {config.detector.enabled}")
print(f"  confidence_threshold: {config.detector.confidence_threshold} (优化前: 0.25)")
print(f"  model_path: {config.detector.model_path}")
print()

print("VLMConfig:")
print(f"  annotated_frames_per_clip: {config.vlm.annotated_frames_per_clip} (优化前: 4)")
print()

print("TemplateConfig:")
print(f"  bike_wrong_way: {len(config.templates.builtin_templates['bike_wrong_way'])}个模板 (优化前: 3)")
print(f"  run_red_light: {len(config.templates.builtin_templates['run_red_light'])}个模板 (优化前: 3)")
print(f"  occupy_motor_lane: {len(config.templates.builtin_templates['occupy_motor_lane'])}个模板 (优化前: 3)")
print(f"  accident: {len(config.templates.builtin_templates['accident'])}个模板 (优化前: 3)")
print()

print("=" * 80)
print("验证结果")
print("=" * 80)
print()

# 检查所有关键参数
checks = [
    (config.stream.motion_min_fg_ratio == 0.015, "motion_min_fg_ratio"),
    (config.stream.motion_debounce_frames == 2, "motion_debounce_frames"),
    (config.embedding.top_m_per_template == 80, "top_m_per_template"),
    (config.embedding.clip_embedding_frames == 10, "clip_embedding_frames"),
    (config.cluster.pre_padding == 4.0, "pre_padding"),
    (config.detector.confidence_threshold == 0.2, "confidence_threshold"),
    (config.vlm.annotated_frames_per_clip == 6, "annotated_frames_per_clip"),
    (len(config.templates.builtin_templates['bike_wrong_way']) >= 7, "bike_wrong_way模板"),
]

all_passed = True
for check, name in checks:
    status = "✓ 通过" if check else "✗ 失败"
    print(f"{status}: {name}")
    if not check:
        all_passed = False

print()
if all_passed:
    print("✓ 所有配置优化已成功应用！")
else:
    print("✗ 部分配置未正确应用，请检查config.py文件")

print()
print("=" * 80)
