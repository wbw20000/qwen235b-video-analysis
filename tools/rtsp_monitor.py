#!/usr/bin/env python3
"""
RTSP 视频流监控脚本

功能：
1. 监控多路 RTSP 视频流
2. 录制视频并分析
3. 每 4 小时生成汇总报告

使用方式：
    python tools/rtsp_monitor.py --config configs/cameras.json
    python tools/rtsp_monitor.py --report-only  # 仅生成报告
"""

import os
import sys
import json
import time
import subprocess
import threading
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 摄像头配置
CAMERAS = [
    {"id": "147-01", "ip": "172.21.14.129"},
    {"id": "147-02", "ip": "172.21.14.130"},
    {"id": "147-03", "ip": "172.21.14.131"},
    {"id": "147-04", "ip": "172.21.14.132"},
    {"id": "147-05", "ip": "172.21.14.133"},
    {"id": "147-06", "ip": "172.21.14.134"},
    {"id": "147-07", "ip": "172.21.14.136"},
    {"id": "147-08", "ip": "172.21.14.137"},
    {"id": "147-09", "ip": "172.21.14.139"},
    {"id": "147-10", "ip": "172.21.14.140"},
    {"id": "147-11", "ip": "172.21.14.141"},
    {"id": "146-01", "ip": "172.21.15.1"},
    {"id": "146-02", "ip": "172.21.15.2"},
    {"id": "146-03", "ip": "172.21.15.3"},
    {"id": "146-04", "ip": "172.21.15.4"},
    {"id": "146-05", "ip": "172.21.15.5"},
    {"id": "146-06", "ip": "172.21.15.7"},
    {"id": "146-07", "ip": "172.21.15.8"},
    {"id": "146-08", "ip": "172.21.15.9"},
    {"id": "146-09", "ip": "172.21.15.12"},
    {"id": "146-10", "ip": "172.21.15.13"},
    {"id": "146-11", "ip": "172.21.15.14"},
    {"id": "146-12", "ip": "172.21.15.15"},
    {"id": "146-13", "ip": "172.21.15.16"},
    {"id": "146-14", "ip": "172.21.15.19"},
    {"id": "146-15", "ip": "172.21.15.20"},
    {"id": "146-16", "ip": "172.21.15.21"},
    {"id": "146-17", "ip": "172.21.15.22"},
    {"id": "146-18", "ip": "172.21.15.23"},
]

RTSP_TEMPLATE = "rtsp://admin:baidu123@{ip}:554/Streaming/Channels/102?transportmode=unicast"
SEGMENT_DURATION = 600  # 10 分钟切片
REPORT_INTERVAL = 4 * 3600  # 4 小时

# API Gateway 配置
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:30500")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


@dataclass
class CameraStatus:
    camera_id: str
    ip: str
    is_online: bool
    last_check: str
    segments_recorded: int = 0
    segments_submitted: int = 0
    accidents_detected: int = 0
    violations_detected: int = 0
    error_message: str = ""


@dataclass
class DiskUsage:
    mount_point: str
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float


@dataclass
class DetectionStats:
    accidents_detected: int = 0
    accident_videos: List[str] = None
    ebike_violations_detected: int = 0
    ebike_violation_videos: List[str] = None

    def __post_init__(self):
        if self.accident_videos is None:
            self.accident_videos = []
        if self.ebike_violation_videos is None:
            self.ebike_violation_videos = []


@dataclass
class MonitorReport:
    report_time: str
    period_start: str
    period_end: str
    total_cameras: int
    online_cameras: int
    offline_cameras: int
    total_segments: int
    total_submitted: int
    total_accidents: int
    total_violations: int
    camera_status: List[Dict]
    # 新增字段
    disk_usage: Dict = None
    detection_stats: Dict = None


class RTSPMonitor:
    """RTSP 视频流监控器"""

    def __init__(self, output_dir: str = "/data1/videos/rtsp_recordings",
                 api_gateway_url: str = None,
                 enable_analysis: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path("/data/app/outputs/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.cameras = CAMERAS
        self.camera_status: Dict[str, CameraStatus] = {}
        self.running = False
        self.threads: List[threading.Thread] = []

        # 分析配置
        self.enable_analysis = enable_analysis
        self.api_gateway_url = api_gateway_url or API_GATEWAY_URL
        self.submit_lock = threading.Lock()
        self.submit_stats = {"success": 0, "failed": 0}

    def get_rtsp_url(self, ip: str) -> str:
        return RTSP_TEMPLATE.format(ip=ip)

    def check_camera_online(self, ip: str) -> bool:
        """检查摄像头是否在线"""
        url = self.get_rtsp_url(ip)
        try:
            # 使用 ffprobe 检查流是否可用
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-rtsp_transport", "tcp",
                 "-i", url, "-show_entries", "stream=codec_type",
                 "-of", "default=noprint_wrappers=1"],
                timeout=10,
                capture_output=True
            )
            return result.returncode == 0
        except Exception as e:
            log.error(f"检查摄像头 {ip} 失败: {e}")
            return False

    def submit_for_analysis(self, video_path: Path, camera_id: str) -> bool:
        """
        提交视频到 API Gateway 进行分析

        Args:
            video_path: 视频文件路径
            camera_id: 摄像头 ID

        Returns:
            是否提交成功
        """
        if not self.enable_analysis:
            return True

        try:
            url = f"{self.api_gateway_url}/api/v1/tasks"
            payload = {
                "video_path": str(video_path),
                "camera_id": camera_id
            }

            response = requests.post(
                url,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job_id", "unknown")
                log.info(f"[{camera_id}] 已提交分析: job_id={job_id}")

                with self.submit_lock:
                    self.submit_stats["success"] += 1
                    self.camera_status[camera_id].segments_submitted += 1

                return True
            else:
                log.warning(f"[{camera_id}] 提交分析失败: HTTP {response.status_code} - {response.text[:200]}")
                with self.submit_lock:
                    self.submit_stats["failed"] += 1
                return False

        except requests.exceptions.Timeout:
            log.warning(f"[{camera_id}] 提交分析超时")
            with self.submit_lock:
                self.submit_stats["failed"] += 1
            return False
        except requests.exceptions.ConnectionError as e:
            log.warning(f"[{camera_id}] 无法连接 API Gateway: {e}")
            with self.submit_lock:
                self.submit_stats["failed"] += 1
            return False
        except Exception as e:
            log.error(f"[{camera_id}] 提交分析异常: {e}")
            with self.submit_lock:
                self.submit_stats["failed"] += 1
            return False

    def record_camera(self, camera_id: str, ip: str):
        """录制单个摄像头"""
        url = self.get_rtsp_url(ip)
        camera_dir = self.output_dir / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)

        while self.running:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = camera_dir / f"{camera_id}_{timestamp}.mp4"

            try:
                log.info(f"[{camera_id}] 开始录制: {output_file}")

                # FFmpeg 录制命令
                cmd = [
                    "ffmpeg", "-y",
                    "-rtsp_transport", "tcp",
                    "-i", url,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-t", str(SEGMENT_DURATION),
                    "-f", "mp4",
                    str(output_file)
                ]

                result = subprocess.run(cmd, timeout=SEGMENT_DURATION + 60,
                                        capture_output=True)

                if result.returncode == 0 and output_file.exists():
                    file_size_mb = output_file.stat().st_size / 1024 / 1024
                    self.camera_status[camera_id].segments_recorded += 1
                    self.camera_status[camera_id].is_online = True
                    log.info(f"[{camera_id}] 录制完成: {file_size_mb:.1f} MB")

                    # 提交到分析队列
                    if self.enable_analysis and file_size_mb > 1:  # 只提交大于1MB的视频
                        self.submit_for_analysis(output_file, camera_id)
                else:
                    self.camera_status[camera_id].is_online = False
                    self.camera_status[camera_id].error_message = result.stderr.decode()[-200:]
                    log.error(f"[{camera_id}] 录制失败，等待 60 秒后重试")
                    time.sleep(60)  # 录制失败等待 1 分钟后重试

            except subprocess.TimeoutExpired:
                log.warning(f"[{camera_id}] 录制超时，等待 60 秒后重试")
                self.camera_status[camera_id].is_online = False
                time.sleep(60)
            except Exception as e:
                log.error(f"[{camera_id}] 录制异常: {e}，等待 60 秒后重试")
                self.camera_status[camera_id].is_online = False
                time.sleep(60)

            self.camera_status[camera_id].last_check = datetime.now().isoformat()

    def get_disk_usage(self) -> Dict[str, DiskUsage]:
        """获取磁盘使用情况"""
        disk_info = {}
        mount_points = ["/data1", "/data", "/"]

        for mount in mount_points:
            try:
                result = subprocess.run(
                    ["df", "-BG", mount],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) >= 2:
                        parts = lines[1].split()
                        if len(parts) >= 4:
                            total = float(parts[1].replace("G", ""))
                            used = float(parts[2].replace("G", ""))
                            free = float(parts[3].replace("G", ""))
                            usage_pct = float(parts[4].replace("%", ""))
                            disk_info[mount] = DiskUsage(
                                mount_point=mount,
                                total_gb=total,
                                used_gb=used,
                                free_gb=free,
                                usage_percent=usage_pct
                            )
            except Exception as e:
                log.warning(f"获取磁盘 {mount} 信息失败: {e}")

        return disk_info

    def get_detection_stats(self, period_start: datetime) -> DetectionStats:
        """获取检测统计（事故和二轮车违法）"""
        stats = DetectionStats()

        # 检查事故检测结果目录
        result_dir = Path("/data/app/data/video_results")
        if result_dir.exists():
            import gzip
            for f in result_dir.glob("*.result.json.gz"):
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime >= period_start:
                        with gzip.open(f, "rt", encoding="utf-8") as gz:
                            data = json.load(gz)
                            verdict = data.get("verdict", "")
                            analysis_type = data.get("analysis_type", "accident")

                            if verdict == "YES":
                                if analysis_type == "ebike_violation":
                                    stats.ebike_violations_detected += 1
                                    stats.ebike_violation_videos.append(f.stem.replace(".result", ""))
                                else:
                                    stats.accidents_detected += 1
                                    stats.accident_videos.append(f.stem.replace(".result", ""))
                except Exception as e:
                    log.debug(f"读取结果文件失败 {f}: {e}")

        # 也检查 Redis 中的结果
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, db=0)
            # 检查 result_tasks stream
            results = r.xrange("result_tasks", "-", "+", count=1000)
            for msg_id, data in results:
                try:
                    created = float(data.get(b"created_at", 0))
                    if datetime.fromtimestamp(created) >= period_start:
                        analysis_type = data.get(b"analysis_type", b"accident").decode()
                        if analysis_type == "ebike_violation":
                            stats.ebike_violations_detected += 1
                        else:
                            stats.accidents_detected += 1
                except:
                    pass
        except Exception as e:
            log.debug(f"读取 Redis 统计失败: {e}")

        return stats

    def generate_report(self) -> MonitorReport:
        """生成汇总报告"""
        now = datetime.now()
        period_start = now - timedelta(hours=4)

        online_count = sum(1 for s in self.camera_status.values() if s.is_online)
        offline_count = len(self.camera_status) - online_count
        total_segments = sum(s.segments_recorded for s in self.camera_status.values())
        total_submitted = sum(s.segments_submitted for s in self.camera_status.values())
        total_accidents = sum(s.accidents_detected for s in self.camera_status.values())
        total_violations = sum(s.violations_detected for s in self.camera_status.values())

        # 获取磁盘使用情况
        disk_usage = self.get_disk_usage()
        disk_usage_dict = {k: asdict(v) for k, v in disk_usage.items()}

        # 获取检测统计
        detection_stats = self.get_detection_stats(period_start)

        report = MonitorReport(
            report_time=now.isoformat(),
            period_start=period_start.isoformat(),
            period_end=now.isoformat(),
            total_cameras=len(self.cameras),
            online_cameras=online_count,
            offline_cameras=offline_count,
            total_segments=total_segments,
            total_submitted=total_submitted,
            total_accidents=total_accidents,
            total_violations=total_violations,
            camera_status=[asdict(s) for s in self.camera_status.values()],
            disk_usage=disk_usage_dict,
            detection_stats=asdict(detection_stats)
        )

        return report

    def save_report(self, report: MonitorReport):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 格式
        json_path = self.report_dir / f"monitor_report_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        # Markdown 格式
        md_path = self.report_dir / f"monitor_report_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# RTSP 监控汇总报告\n\n")
            f.write(f"**报告时间**: {report.report_time}\n\n")
            f.write(f"**统计周期**: {report.period_start} ~ {report.period_end}\n\n")

            # 磁盘使用情况
            f.write(f"## 1. 磁盘使用情况\n\n")
            if report.disk_usage:
                f.write(f"| 挂载点 | 总容量 | 已使用 | 剩余 | 使用率 |\n")
                f.write(f"|--------|--------|--------|------|--------|\n")
                for mount, disk in report.disk_usage.items():
                    f.write(f"| {mount} | {disk['total_gb']:.1f} GB | {disk['used_gb']:.1f} GB | {disk['free_gb']:.1f} GB | {disk['usage_percent']:.1f}% |\n")
            else:
                f.write("无法获取磁盘信息\n")

            # 提交统计
            f.write(f"\n## 2. 视频提交统计\n\n")
            f.write(f"| 指标 | 数值 |\n")
            f.write(f"|------|------|\n")
            f.write(f"| 录制片段 | {report.total_segments} |\n")
            f.write(f"| 已提交分析 | {report.total_submitted} |\n")
            with self.submit_lock:
                f.write(f"| 提交成功 | {self.submit_stats['success']} |\n")
                f.write(f"| 提交失败 | {self.submit_stats['failed']} |\n")

            # 事故检测统计
            f.write(f"\n## 3. 事故检测统计\n\n")
            if report.detection_stats:
                accidents = report.detection_stats.get("accidents_detected", 0)
                accident_videos = report.detection_stats.get("accident_videos", [])
                if accidents > 0:
                    f.write(f"**✅ 检测到 {accidents} 起事故**\n\n")
                    if accident_videos:
                        f.write("| 事故视频 |\n")
                        f.write("|----------|\n")
                        for v in accident_videos[:20]:  # 最多显示20条
                            f.write(f"| {v} |\n")
                        if len(accident_videos) > 20:
                            f.write(f"| ... 共 {len(accident_videos)} 条 |\n")
                else:
                    f.write("**本周期内未检测到事故**\n")
            else:
                f.write("无检测数据\n")

            # 二轮车违法检测统计
            f.write(f"\n## 4. 二轮车违法检测统计\n\n")
            if report.detection_stats:
                violations = report.detection_stats.get("ebike_violations_detected", 0)
                violation_videos = report.detection_stats.get("ebike_violation_videos", [])
                if violations > 0:
                    f.write(f"**✅ 检测到 {violations} 起二轮车违法**\n\n")
                    if violation_videos:
                        f.write("| 违法视频 |\n")
                        f.write("|----------|\n")
                        for v in violation_videos[:20]:
                            f.write(f"| {v} |\n")
                        if len(violation_videos) > 20:
                            f.write(f"| ... 共 {len(violation_videos)} 条 |\n")
                else:
                    f.write("**本周期内未检测到二轮车违法**\n")
            else:
                f.write("无检测数据\n")

            # 监控概览
            f.write(f"\n## 5. 监控概览\n\n")
            f.write(f"| 指标 | 数值 |\n")
            f.write(f"|------|------|\n")
            f.write(f"| 总摄像头 | {report.total_cameras} |\n")
            f.write(f"| 在线 | {report.online_cameras} |\n")
            f.write(f"| 离线 | {report.offline_cameras} |\n")
            f.write(f"| 录制片段 | {report.total_segments} |\n")
            f.write(f"| 已提交分析 | {report.total_submitted} |\n")

            # 摄像头状态
            f.write(f"\n## 6. 摄像头状态\n\n")
            f.write(f"| 摄像头 | IP | 状态 | 录制 | 提交 |\n")
            f.write(f"|--------|-----|------|------|------|\n")
            for s in report.camera_status:
                status = "✅ 在线" if s["is_online"] else "❌ 离线"
                f.write(f"| {s['camera_id']} | {s['ip']} | {status} | {s['segments_recorded']} | {s.get('segments_submitted', 0)} |\n")

            f.write(f"\n---\n*报告自动生成于 {report.report_time}*\n")

        log.info(f"报告已保存: {json_path}")
        log.info(f"报告已保存: {md_path}")

        return json_path, md_path

    def report_loop(self):
        """定时报告循环"""
        while self.running:
            time.sleep(REPORT_INTERVAL)
            if self.running:
                report = self.generate_report()
                self.save_report(report)

    def start(self):
        """启动监控"""
        log.info(f"启动 RTSP 监控，共 {len(self.cameras)} 路摄像头")
        log.info(f"分析功能: {'启用' if self.enable_analysis else '禁用'}")
        if self.enable_analysis:
            log.info(f"API Gateway: {self.api_gateway_url}")
        self.running = True

        # 初始化摄像头状态
        for cam in self.cameras:
            self.camera_status[cam["id"]] = CameraStatus(
                camera_id=cam["id"],
                ip=cam["ip"],
                is_online=False,
                last_check=datetime.now().isoformat()
            )

        # 启动录制线程
        for cam in self.cameras:
            t = threading.Thread(target=self.record_camera, args=(cam["id"], cam["ip"]))
            t.daemon = True
            t.start()
            self.threads.append(t)
            time.sleep(0.5)  # 错开启动时间

        # 启动报告线程
        report_thread = threading.Thread(target=self.report_loop)
        report_thread.daemon = True
        report_thread.start()
        self.threads.append(report_thread)

        log.info("所有监控线程已启动")

    def stop(self):
        """停止监控"""
        log.info("停止监控...")
        self.running = False

        # 生成最终报告
        report = self.generate_report()
        self.save_report(report)

        log.info("监控已停止")

    def check_all_cameras(self):
        """检查所有摄像头状态"""
        log.info("检查摄像头在线状态...")
        online_count = 0
        for cam in self.cameras:
            is_online = self.check_camera_online(cam["ip"])
            status = "✅ 在线" if is_online else "❌ 离线"
            log.info(f"  {cam['id']} ({cam['ip']}): {status}")
            if is_online:
                online_count += 1
        log.info(f"在线摄像头: {online_count}/{len(self.cameras)}")
        return online_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RTSP 视频流监控')
    parser.add_argument('--check', action='store_true', help='仅检查摄像头在线状态')
    parser.add_argument('--report-only', action='store_true', help='仅生成报告')
    parser.add_argument('--output-dir', default='/data1/videos/rtsp_recordings', help='录制输出目录')
    parser.add_argument('--no-analysis', action='store_true', help='禁用分析提交（仅录制）')
    parser.add_argument('--api-gateway', default=None, help='API Gateway URL')
    args = parser.parse_args()

    monitor = RTSPMonitor(
        output_dir=args.output_dir,
        api_gateway_url=args.api_gateway,
        enable_analysis=not args.no_analysis
    )

    if args.check:
        monitor.check_all_cameras()
        return

    if args.report_only:
        report = monitor.generate_report()
        monitor.save_report(report)
        return

    try:
        monitor.start()
        # 保持运行
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main()
