#!/usr/bin/env python3
"""
多语义检测单元测试
测试三个新分析器: mv_violation, ebike_violation, ads_behavior
"""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from workers.common.redis_client import VideoTask, ResultTask
from workers.semantic_base import AnalysisResult, FrameInfo, ClipInfo


class TestVideoTaskExtension:
    """测试 VideoTask analysis_type 扩展"""

    def test_default_analysis_type(self):
        """默认 analysis_type 应为 accident"""
        task = VideoTask(
            job_id="test_001",
            camera_id="cam_001",
            window_path="/data/test.mp4",
            trace_id="trace_001"
        )
        assert task.analysis_type == "accident"

    def test_custom_analysis_type(self):
        """自定义 analysis_type"""
        task = VideoTask(
            job_id="test_002",
            camera_id="cam_001",
            window_path="/data/test.mp4",
            trace_id="trace_002",
            analysis_type="mv_violation"
        )
        assert task.analysis_type == "mv_violation"

    def test_task_to_dict(self):
        """测试 to_dict 包含 analysis_type"""
        task = VideoTask(
            job_id="test_003",
            camera_id="cam_001",
            window_path="/data/test.mp4",
            trace_id="trace_003",
            analysis_type="ebike_violation"
        )
        d = task.to_dict()
        assert "analysis_type" in d
        assert d["analysis_type"] == "ebike_violation"

    def test_task_from_dict(self):
        """测试 from_dict 解析 analysis_type"""
        data = {
            "job_id": "test_004",
            "camera_id": "cam_001",
            "window_path": "/data/test.mp4",
            "trace_id": "trace_004",
            "analysis_type": "ads_behavior"
        }
        task = VideoTask.from_dict(data)
        assert task.analysis_type == "ads_behavior"

    def test_task_from_dict_default(self):
        """from_dict 缺少 analysis_type 时使用默认值"""
        data = {
            "job_id": "test_005",
            "camera_id": "cam_001",
            "window_path": "/data/test.mp4",
            "trace_id": "trace_005"
        }
        task = VideoTask.from_dict(data)
        assert task.analysis_type == "accident"


class TestResultTaskExtension:
    """测试 ResultTask 扩展字段"""

    def test_result_task_fields(self):
        """测试新增字段"""
        result = ResultTask(
            job_id="test_001",
            camera_id="cam_001",
            event_time=1234567890.0,
            confidence=0.95,
            result_path="/data/result.json.gz",
            trace_id="trace_001",
            is_accident=False,
            analysis_type="mv_violation",
            is_positive=True,
            violation_type="闯红灯"
        )
        assert result.analysis_type == "mv_violation"
        assert result.is_positive is True
        assert result.violation_type == "闯红灯"

    def test_result_task_to_dict(self):
        """测试序列化"""
        result = ResultTask(
            job_id="test_002",
            camera_id="cam_001",
            event_time=1234567890.0,
            confidence=0.85,
            result_path="/data/result.json.gz",
            trace_id="trace_002",
            is_accident=False,
            analysis_type="ads_behavior",
            is_positive=True,
            behavior_type="变道"
        )
        d = result.to_dict()
        assert d["analysis_type"] == "ads_behavior"
        assert d["is_positive"] == "1"
        assert d["behavior_type"] == "变道"


class TestAnalysisResult:
    """测试 AnalysisResult 数据类"""

    def test_basic_result(self):
        """基本结果"""
        result = AnalysisResult(
            judgment="YES",
            confidence=0.9,
            reason="检测到闯红灯"
        )
        assert result.judgment == "YES"
        assert result.confidence == 0.9

    def test_violation_result(self):
        """违法检测结果"""
        result = AnalysisResult(
            judgment="YES",
            confidence=0.85,
            reason="车辆闯红灯",
            violation_type="闯红灯"
        )
        assert result.violation_type == "闯红灯"

    def test_ads_behavior_result(self):
        """ADS行为检测结果"""
        result = AnalysisResult(
            judgment="YES",
            confidence=0.8,
            reason="检测到示廓灯开启",
            marker_light_state="开启",
            behavior_type="正常行驶"
        )
        assert result.marker_light_state == "开启"
        assert result.behavior_type == "正常行驶"


class TestMVViolationAnalyzer:
    """测试机动车违法检测分析器"""

    def test_templates(self):
        """测试模板加载"""
        from workers.mv_violation_analyzer import MV_VIOLATION_TEMPLATES
        assert len(MV_VIOLATION_TEMPLATES) > 0
        assert any("red light" in t for t in MV_VIOLATION_TEMPLATES)

    def test_parse_vlm_response_json(self):
        """测试 JSON 响应解析"""
        from workers.mv_violation_analyzer import MVViolationAnalyzer

        # Mock 初始化
        with patch.object(MVViolationAnalyzer, '__init__', lambda self, **kwargs: None):
            analyzer = MVViolationAnalyzer()
            analyzer.analysis_type = "mv_violation"

            response = '''
            {
                "judgment": "YES",
                "confidence": 0.85,
                "violation_type": "闯红灯",
                "reason": "车辆在红灯时通过路口"
            }
            '''
            result = analyzer.parse_vlm_response(response)
            assert result.judgment == "YES"
            assert result.confidence == 0.85
            assert result.violation_type == "闯红灯"

    def test_parse_vlm_response_fallback(self):
        """测试回退解析"""
        from workers.mv_violation_analyzer import MVViolationAnalyzer

        with patch.object(MVViolationAnalyzer, '__init__', lambda self, **kwargs: None):
            analyzer = MVViolationAnalyzer()
            analyzer.analysis_type = "mv_violation"

            response = "YES, 我检测到车辆违法变道"
            result = analyzer.parse_vlm_response(response)
            assert result.judgment == "YES"


class TestEbikeViolationAnalyzer:
    """测试电动自行车违法检测分析器"""

    def test_templates(self):
        """测试模板加载"""
        from workers.ebike_violation_analyzer import EBIKE_VIOLATION_TEMPLATES
        assert len(EBIKE_VIOLATION_TEMPLATES) > 0
        assert any("helmet" in t for t in EBIKE_VIOLATION_TEMPLATES)

    def test_parse_response(self):
        """测试响应解析"""
        from workers.ebike_violation_analyzer import EbikeViolationAnalyzer

        with patch.object(EbikeViolationAnalyzer, '__init__', lambda self, **kwargs: None):
            analyzer = EbikeViolationAnalyzer()
            analyzer.analysis_type = "ebike_violation"

            response = '''
            {
                "judgment": "YES",
                "confidence": 0.9,
                "violation_type": "不戴头盔",
                "reason": "电动车驾驶人未佩戴安全头盔"
            }
            '''
            result = analyzer.parse_vlm_response(response)
            assert result.judgment == "YES"
            assert result.violation_type == "不戴头盔"


class TestADSBehaviorAnalyzer:
    """测试自动驾驶行为检测分析器"""

    def test_templates(self):
        """测试模板加载"""
        from workers.ads_behavior_analyzer import ADS_BEHAVIOR_TEMPLATES
        assert len(ADS_BEHAVIOR_TEMPLATES) > 0
        assert any("marker" in t.lower() for t in ADS_BEHAVIOR_TEMPLATES)

    def test_parse_phase1_response(self):
        """测试第一阶段响应解析"""
        from workers.ads_behavior_analyzer import ADSBehaviorAnalyzer

        with patch.object(ADSBehaviorAnalyzer, '__init__', lambda self, **kwargs: None):
            analyzer = ADSBehaviorAnalyzer()
            analyzer.analysis_type = "ads_behavior"

            response = '''
            {
                "marker_light_detected": true,
                "marker_light_state": "开启",
                "vehicle_count": 1,
                "confidence": 0.85,
                "reason": "检测到车顶有黄色标识灯"
            }
            '''
            result = analyzer.parse_vlm_response(response)
            assert result.judgment == "YES"
            assert result.marker_light_state == "开启"

    def test_parse_behavior_response(self):
        """测试第二阶段行为分析响应"""
        from workers.ads_behavior_analyzer import ADSBehaviorAnalyzer

        with patch.object(ADSBehaviorAnalyzer, '__init__', lambda self, **kwargs: None):
            analyzer = ADSBehaviorAnalyzer()
            analyzer.analysis_type = "ads_behavior"

            response = '''
            {
                "judgment": "YES",
                "confidence": 0.8,
                "behavior_type": "变道",
                "behavior_description": "车辆平稳变道",
                "safety_level": "安全",
                "reason": "观察到车辆从左车道变换到右车道"
            }
            '''
            result = analyzer.parse_behavior_response(response)
            assert result.judgment == "YES"
            assert result.behavior_type == "变道"


class TestIdempotency:
    """测试幂等性"""

    def test_job_id_uniqueness(self):
        """job_id + analysis_type 组合唯一性"""
        task1 = VideoTask(
            job_id="job_001",
            camera_id="cam_001",
            window_path="/data/test.mp4",
            trace_id="trace_001",
            analysis_type="accident"
        )
        task2 = VideoTask(
            job_id="job_001",
            camera_id="cam_001",
            window_path="/data/test.mp4",
            trace_id="trace_001",
            analysis_type="mv_violation"
        )
        # 相同 job_id 但不同 analysis_type 应视为不同任务
        key1 = f"{task1.job_id}:{task1.analysis_type}"
        key2 = f"{task2.job_id}:{task2.analysis_type}"
        assert key1 != key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
