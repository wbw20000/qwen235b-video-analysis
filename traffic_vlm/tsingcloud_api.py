"""
云控智行 API 客户端

提供历史视频查询与下载功能：
1. Token认证（SHA256签名，7天有效期）
2. 获取路口摄像头列表
3. 获取视频下载URL（支持轮询等待）
4. RTSP流直接下载（更快，无需等待生成）

双账号模式：
- HTTP轮询账号: 用于 /v2x/platform/device/road/info 接口
- RTSP账号: 用于 /monitorModel/singleCameraQuery2 接口
"""
from __future__ import annotations

import hashlib
import time
import logging
import requests
import subprocess
import os
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable

# 禁用SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class DeviceMapper:
    """
    设备映射器：将roadId映射到deviceId（用于RTSP下载）

    设备映射文件格式：deviceId,sn,deviceCate,roadId
    - deviceId: 设备ID（如 N-BD003H），用于singleCameraQuery2 API
    - deviceCate: DJ=电警(全景), KK=卡口(抓拍)
    """

    def __init__(self, mapping_file: str):
        self.mapping_file = mapping_file
        self._cache: Dict[str, List[Dict[str, str]]] = {}  # {roadId: [{deviceId, sn, cate}, ...]}
        self._loaded = False

    def _load_mapping(self):
        """加载设备映射文件（保留所有设备）"""
        if self._loaded:
            return

        if not os.path.exists(self.mapping_file):
            logger.warning(f"设备映射文件不存在: {self.mapping_file}")
            self._loaded = True
            return

        try:
            with open(self.mapping_file, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    road_id = row.get('roadId', '').strip()
                    device_cate = row.get('deviceCate', '').strip()
                    device_id = row.get('deviceId', '').strip()
                    sn = row.get('sn', '').strip()

                    if road_id and device_id:
                        if road_id not in self._cache:
                            self._cache[road_id] = []
                        self._cache[road_id].append({
                            'deviceId': device_id,
                            'sn': sn,
                            'cate': device_cate
                        })

            total_devices = sum(len(v) for v in self._cache.values())
            logger.info(f"设备映射加载完成: {len(self._cache)} 个路口, {total_devices} 个设备")
            self._loaded = True

        except Exception as e:
            logger.error(f"加载设备映射文件失败: {e}")
            self._loaded = True

    def get_device_id(self, road_id: str, device_cate: str = "DJ", index: int = 0) -> Optional[str]:
        """
        获取路口对应的deviceId

        Args:
            road_id: 路口ID
            device_cate: 设备类型 "DJ"(全景) 或 "KK"(抓拍)，默认DJ
            index: 设备索引（同类型设备中的第几个，从0开始）

        Returns:
            deviceId（如 N-BD003H），未找到返回None
        """
        self._load_mapping()
        devices = self._cache.get(str(road_id), [])

        # 筛选指定类型的设备
        filtered = [d for d in devices if d['cate'] == device_cate]

        if index < len(filtered):
            return filtered[index]['deviceId']
        elif filtered:
            # 索引超出范围，返回第一个
            return filtered[0]['deviceId']
        return None

    def get_all_devices(self, road_id: str, device_cate: str = None) -> List[Dict[str, str]]:
        """
        获取路口的所有设备

        Args:
            road_id: 路口ID
            device_cate: 可选，筛选设备类型 "DJ" 或 "KK"

        Returns:
            设备列表 [{deviceId, sn, cate}, ...]
        """
        self._load_mapping()
        devices = self._cache.get(str(road_id), [])

        if device_cate:
            return [d for d in devices if d['cate'] == device_cate]
        return devices

    def get_device_info(self, road_id: str, device_cate: str = "DJ", index: int = 0) -> Optional[Dict[str, str]]:
        """获取路口对应的设备信息"""
        self._load_mapping()
        devices = self._cache.get(str(road_id), [])
        filtered = [d for d in devices if d['cate'] == device_cate]

        if index < len(filtered):
            return filtered[index]
        elif filtered:
            return filtered[0]
        return None


@dataclass
class CameraInfo:
    """摄像头信息"""
    channel_num: str          # 摄像头通道号
    camera_type: int          # 0=抓拍摄像头, 1=全景摄像头
    camera_type_str: str      # 类型描述
    request_id: str           # 用于获取视频URL的requestId
    sn: str                   # 序列号
    cam_std_code: str         # 标准代码

    @property
    def is_panoramic(self) -> bool:
        """是否为全景摄像头"""
        return self.camera_type == 1

    @classmethod
    def from_dict(cls, data: dict) -> "CameraInfo":
        return cls(
            channel_num=data.get("channelNum", ""),
            camera_type=data.get("cameraType", 0),
            camera_type_str=data.get("cameraTypeStr", ""),
            request_id=data.get("requestId", ""),
            sn=data.get("sn", ""),
            cam_std_code=data.get("camStdCode", "")
        )


class TsingcloudAPIError(Exception):
    """云控智行API错误"""
    def __init__(self, message: str, code: int = -1, response: dict = None):
        super().__init__(message)
        self.code = code
        self.response = response or {}


class TsingcloudAPI:
    """
    云控智行 API 客户端

    Usage:
        api = TsingcloudAPI(app_key="xxx", password="xxx")
        cameras = api.get_road_cameras("1", "20241215100000", "20241215100500")
        video_url = api.get_video_url("1", cameras[0].request_id)
    """

    DEFAULT_BASE_URL = "https://rc.ccg.bcavt.com:8760/infraCloud"

    # 视频生成状态码（参考API文档第8页）
    STATUS_OK = 0              # 成功
    STATUS_GENERATING = 30056  # URL生成中，请30秒/次查询结果
    STATUS_REQUEST_ID_EMPTY = 30054  # 请求id为空
    STATUS_HISTORY_VIDEO_FAILED = 30055  # 查询历史视频失败，requestId错误
    STATUS_RATE_LIMITED = 50001  # 访问太频繁，超过限流控制

    def __init__(
        self,
        app_key: str,
        password: str,
        base_url: str = None,
        request_interval: float = 1.0,
        poll_interval: float = 30.0,
        poll_timeout: float = 300.0,
        verify_ssl: bool = False
    ):
        """
        初始化API客户端

        Args:
            app_key: 应用Key（用户名）
            password: 密码
            base_url: API基础URL
            request_interval: 请求间隔（秒），防止限流
            poll_interval: 轮询间隔（秒）
            poll_timeout: 轮询超时（秒）
            verify_ssl: 是否验证SSL证书
        """
        self.app_key = app_key
        self.password = password
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.request_interval = request_interval
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.verify_ssl = verify_ssl

        # Token缓存
        self._token: Optional[str] = None
        self._token_expires_at: float = 0

        # 最后请求时间（用于限流）
        self._last_request_time: float = 0

    def _generate_sign(self, timestamp: int) -> str:
        """生成SHA256签名"""
        sign_str = f"{self.app_key}_{timestamp}_{self.password}"
        return hashlib.sha256(sign_str.encode()).hexdigest()

    def _wait_for_rate_limit(self):
        """等待以遵守请求频率限制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        json_data: dict = None,
        headers: dict = None,
        timeout: int = 30
    ) -> dict:
        """发送HTTP请求"""
        self._wait_for_rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = headers or {}

        try:
            if method.upper() == "GET":
                response = requests.get(
                    url, params=params, headers=headers,
                    verify=self.verify_ssl, timeout=timeout
                )
            else:
                response = requests.post(
                    url, json=json_data, headers=headers,
                    verify=self.verify_ssl, timeout=timeout
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise TsingcloudAPIError("请求超时", code=-1)
        except requests.exceptions.RequestException as e:
            raise TsingcloudAPIError(f"请求失败: {e}", code=-1)

    def get_token(self, force_refresh: bool = False) -> str:
        """
        获取认证Token（带缓存）

        Args:
            force_refresh: 强制刷新Token

        Returns:
            Token字符串
        """
        # 检查缓存（预留5分钟buffer）
        if not force_refresh and self._token and time.time() < self._token_expires_at - 300:
            return self._token

        timestamp = int(time.time() * 1000)
        sign = self._generate_sign(timestamp)

        result = self._request(
            "POST",
            "/auth/token",
            json_data={
                "appKey": self.app_key,
                "timestamp": timestamp,
                "sign": sign
            }
        )

        if result.get("code") != 0:
            raise TsingcloudAPIError(
                f"获取Token失败: {result.get('message', '未知错误')}",
                code=result.get("code", -1),
                response=result
            )

        data = result.get("data", {})
        self._token = data.get("token")
        # Token有效期（秒），默认7天
        expires_in = data.get("expired", 604800)
        self._token_expires_at = time.time() + expires_in

        logger.info(f"Token获取成功，有效期: {expires_in}秒")
        return self._token

    def get_road_cameras(
        self,
        road_id: str,
        start_time: str,
        end_time: str
    ) -> List[CameraInfo]:
        """
        获取路口的摄像头列表

        Args:
            road_id: 路口ID
            start_time: 开始时间 (yyyyMMddHHmmss)
            end_time: 结束时间 (yyyyMMddHHmmss)

        Returns:
            摄像头信息列表
        """
        token = self.get_token()

        result = self._request(
            "GET",
            "/openapi/regionCloud/v2x/platform/device/road/info",
            params={
                "roadId": road_id,
                "startTime": start_time,
                "endTime": end_time,
                "videoType": 2
            },
            headers={"token": token}
        )

        status = result.get("status")
        if status != 0:
            raise TsingcloudAPIError(
                f"获取摄像头列表失败: {result.get('msg', '未知错误')}",
                code=status,
                response=result
            )

        cameras = []
        for item in result.get("data", []):
            cameras.append(CameraInfo.from_dict(item))

        logger.info(f"路口{road_id}获取到{len(cameras)}个摄像头")
        return cameras

    def get_video_url(
        self,
        road_id: str,
        request_id: str,
        progress_callback: Callable[[int, int, str], None] = None
    ) -> str:
        """
        获取视频下载URL（支持轮询等待）

        Args:
            road_id: 路口ID
            request_id: 从get_road_cameras获取的requestId
            progress_callback: 进度回调函数 (attempt, max_attempts, message)

        Returns:
            视频下载URL

        Raises:
            TsingcloudAPIError: 获取失败或超时
        """
        token = self.get_token()
        max_attempts = int(self.poll_timeout / self.poll_interval)

        for attempt in range(max_attempts):
            result = self._request(
                "GET",
                "/openapi/regionCloud/v2x/platform/device/road/info",
                params={
                    "requestId": request_id,
                    "roadId": road_id,
                    "videoType": 3
                },
                headers={"token": token}
            )

            status = result.get("status")
            msg = result.get("msg", "")
            data = result.get("data")

            if progress_callback:
                progress_callback(attempt + 1, max_attempts, f"状态: {msg}")

            if status == self.STATUS_OK:
                # 状态OK，检查data是否有效
                if data:
                    video_url = data if isinstance(data, str) else data.get("url", "")
                    # 校验 URL 有效性（API 可能返回 "null" 字符串）
                    if video_url and video_url.lower() not in ('null', 'none', ''):
                        if video_url.startswith(('http://', 'https://')):
                            logger.info(f"视频URL获取成功: {video_url[:50]}...")
                            return video_url
                        else:
                            logger.warning(f"视频URL格式异常: {video_url}")

                # status=0 但 data 为空或无效URL，继续轮询等待
                # 这是正常情况：服务器接受了请求，但视频URL还未生成完成
                logger.debug(f"状态OK但URL未就绪 (data={data})，{self.poll_interval}秒后重试 ({attempt+1}/{max_attempts})")
                if progress_callback:
                    progress_callback(attempt + 1, max_attempts, f"服务器正在生成视频... (尝试 {attempt+1}/{max_attempts})")

                # 分段睡眠，每5秒发送一次心跳回调
                heartbeat_interval = 5
                elapsed = 0
                while elapsed < self.poll_interval:
                    time.sleep(heartbeat_interval)
                    elapsed += heartbeat_interval
                    if progress_callback:
                        progress_callback(attempt + 1, max_attempts, f"等待URL生成... ({elapsed}s/{int(self.poll_interval)}s)")

            elif status == self.STATUS_GENERATING:
                # 视频生成中（状态码30056），分段等待并发送心跳
                logger.debug(f"视频生成中(30056)，{self.poll_interval}秒后重试 ({attempt+1}/{max_attempts})")
                # 分段睡眠，每5秒发送一次心跳回调
                heartbeat_interval = 5
                elapsed = 0
                while elapsed < self.poll_interval:
                    time.sleep(heartbeat_interval)
                    elapsed += heartbeat_interval
                    if progress_callback:
                        progress_callback(attempt + 1, max_attempts, f"等待视频生成中... ({elapsed}s/{int(self.poll_interval)}s)")

            elif status == self.STATUS_HISTORY_VIDEO_FAILED:
                # 30055: 查询历史视频失败 - requestId可能还未生效，等待后重试
                logger.warning(f"历史视频查询失败(30055)，{self.poll_interval}秒后重试 ({attempt+1}/{max_attempts})")
                if progress_callback:
                    progress_callback(attempt + 1, max_attempts, f"视频文件生成中... (尝试 {attempt+1}/{max_attempts})")

                # 分段睡眠
                heartbeat_interval = 5
                elapsed = 0
                while elapsed < self.poll_interval:
                    time.sleep(heartbeat_interval)
                    elapsed += heartbeat_interval
                    if progress_callback:
                        progress_callback(attempt + 1, max_attempts, f"等待服务器处理... ({elapsed}s/{int(self.poll_interval)}s)")

            elif status == self.STATUS_RATE_LIMITED:
                # 50001: 访问太频繁 - 等待更长时间后重试
                logger.warning(f"API限流(50001)，等待60秒后重试")
                if progress_callback:
                    progress_callback(attempt + 1, max_attempts, "API限流，等待60秒...")
                time.sleep(60)

            else:
                # 其他错误状态码 - 立即失败
                raise TsingcloudAPIError(
                    f"获取视频URL失败: {msg} (状态码: {status})",
                    code=status,
                    response=result
                )

        raise TsingcloudAPIError(
            f"获取视频URL超时（已等待{self.poll_timeout}秒，共{max_attempts}次尝试）",
            code=-1
        )

    def get_video_url_for_segment(
        self,
        road_id: str,
        channel_num: str,
        start_time: datetime,
        end_time: datetime,
        progress_callback: Callable[[int, int, str], None] = None
    ) -> str:
        """
        获取指定时间段的视频URL（便捷方法）

        Args:
            road_id: 路口ID
            channel_num: 摄像头通道号
            start_time: 开始时间
            end_time: 结束时间
            progress_callback: 进度回调

        Returns:
            视频下载URL
        """
        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str = end_time.strftime("%Y%m%d%H%M%S")

        # 获取该时间段的摄像头列表（包含requestId）
        cameras = self.get_road_cameras(road_id, start_str, end_str)

        # 找到指定的摄像头
        target_camera = None
        for cam in cameras:
            if cam.channel_num == channel_num:
                target_camera = cam
                break

        if not target_camera:
            raise TsingcloudAPIError(
                f"未找到摄像头: {channel_num}",
                code=-1
            )

        # 获取视频URL
        return self.get_video_url(road_id, target_camera.request_id, progress_callback)

    def get_rtsp_url(
        self,
        device_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[str]:
        """
        获取RTSP流URL（singleCameraQuery2接口，更快速）

        Args:
            device_id: 设备ID（摄像头deviceId）
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            RTSP流URL，失败返回None
        """
        token = self.get_token()

        # 转换为毫秒时间戳
        begin_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)

        payload = {
            "deviceId": device_id,
            "beginTime": begin_time_ms,
            "endTime": end_time_ms
        }

        try:
            result = self._request(
                "POST",
                "/openapi/regionCloud/monitorModel/singleCameraQuery2",
                json_data=payload,
                headers={"token": token, "Content-Type": "application/json"}
            )

            if result.get('code') == 0 and result.get('data', {}).get('thisOne'):
                this_one = result['data']['thisOne']
                history_dtos = this_one.get('historyRtspUrlDTOs', [])

                if history_dtos and history_dtos[0].get('code') == '0':
                    rtsp_url = history_dtos[0].get('rtspUrl')
                    if rtsp_url:
                        logger.info(f"RTSP URL获取成功: {rtsp_url[:50]}...")
                        return rtsp_url

                error_msg = history_dtos[0].get('message', '无有效RTSP URL') if history_dtos else '无历史DTO记录'
                logger.warning(f"RTSP URL获取失败: {error_msg}")
                return None
            else:
                logger.warning(f"RTSP API返回错误: {result.get('message', '未知错误')}")
                return None

        except Exception as e:
            logger.warning(f"RTSP URL获取异常: {e}")
            return None

    def get_device_id_for_camera(
        self,
        road_id: str,
        start_time: str,
        end_time: str,
        camera_type: int = 1
    ) -> Optional[str]:
        """
        获取摄像头的deviceId（用于RTSP下载）

        Args:
            road_id: 路口ID
            start_time: 开始时间 (yyyyMMddHHmmss)
            end_time: 结束时间 (yyyyMMddHHmmss)
            camera_type: 摄像头类型, 1=全景摄像头

        Returns:
            设备ID，失败返回None
        """
        try:
            cameras = self.get_road_cameras(road_id, start_time, end_time)
            for cam in cameras:
                if cam.camera_type == camera_type:
                    # 从sn字段提取deviceId（如果有）
                    if cam.sn:
                        return cam.sn
                    # 或者使用channel_num
                    if cam.channel_num:
                        return cam.channel_num
            return None
        except Exception as e:
            logger.warning(f"获取设备ID失败: {e}")
            return None


def download_video_rtsp(
    rtsp_url: str,
    output_path: str,
    timeout: int = None,  # 取消超时限制，默认无限等待
    progress_callback: Callable[[int, int], None] = None,
    progress_interval: float = 5.0  # 进度显示间隔（秒）
) -> bool:
    """
    使用FFmpeg从RTSP流下载视频（无超时限制，显示实时进度）

    Args:
        rtsp_url: RTSP流URL
        output_path: 输出文件路径
        timeout: 已废弃，不再使用
        progress_callback: 下载进度回调函数 (downloaded_bytes, elapsed_seconds)
        progress_interval: 进度显示间隔（秒）

    Returns:
        是否成功
    """
    import time
    import threading

    if not rtsp_url:
        logger.error("RTSP URL为空")
        return False

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        "-reconnect", "1",
        "-reconnect_delay_max", "10",
        "-y",
        output_path
    ]

    start_time = time.time()
    last_log_time = start_time
    process = None
    stop_monitor = threading.Event()

    def monitor_progress():
        """后台线程：监控下载进度"""
        nonlocal last_log_time
        while not stop_monitor.is_set():
            time.sleep(1)  # 每秒检查一次

            if stop_monitor.is_set():
                break

            current_time = time.time()
            elapsed = current_time - start_time

            # 每 progress_interval 秒输出一次进度
            if current_time - last_log_time >= progress_interval:
                last_log_time = current_time

                if os.path.exists(output_path):
                    try:
                        file_size = os.path.getsize(output_path)
                        size_mb = file_size / (1024 * 1024)

                        # 计算下载速度
                        speed_mbps = size_mb / elapsed if elapsed > 0 else 0

                        logger.info(
                            f"[RTSP下载进度] 已用时: {elapsed:.0f}秒, "
                            f"已下载: {size_mb:.1f}MB, "
                            f"速度: {speed_mbps:.2f}MB/s"
                        )

                        if progress_callback:
                            progress_callback(file_size, int(elapsed))
                    except:
                        pass

    try:
        if progress_callback:
            progress_callback(0, 0)  # 开始信号

        logger.info(f"[RTSP下载开始] 无超时限制，实时显示进度...")

        # 启动FFmpeg进程
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        # 启动进度监控线程
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        # 等待进程完成（无超时）
        stdout, stderr = process.communicate()

        # 停止监控线程
        stop_monitor.set()
        monitor_thread.join(timeout=2)

        elapsed_total = time.time() - start_time

        # 检查返回码和文件
        if process.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                size_mb = file_size / (1024 * 1024)
                if progress_callback:
                    progress_callback(file_size, int(elapsed_total))
                logger.info(
                    f"[RTSP下载完成] {output_path} "
                    f"({size_mb:.1f}MB, 用时 {elapsed_total:.0f}秒)"
                )
                return True
            else:
                logger.error(f"RTSP下载后文件不存在或为空: {output_path}")
                return False
        else:
            error_lines = stderr.strip().split('\n') if stderr else []
            error_msg = error_lines[-1] if error_lines else "未知错误"
            logger.error(f"FFmpeg执行失败 (返回码 {process.returncode}): {error_msg}")
            return False

    except FileNotFoundError:
        logger.error("FFmpeg未安装或不在PATH中")
        stop_monitor.set()
        return False
    except Exception as e:
        logger.error(f"RTSP下载异常: {e}")
        stop_monitor.set()
        # 清理进程
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        return False


def download_video(
    url: str,
    output_path: str,
    timeout: int = 300,
    progress_callback: Callable[[int, int], None] = None
) -> bool:
    """
    下载视频文件

    Args:
        url: 视频URL
        output_path: 输出文件路径
        timeout: 下载超时（秒）
        progress_callback: 下载进度回调函数 (downloaded_bytes, total_bytes)

    Returns:
        是否成功
    """
    import threading

    # URL 有效性校验
    if not url or url == 'null' or url.lower() == 'none':
        logger.error(f"视频下载失败: 无效的 URL '{url}'")
        raise TsingcloudAPIError(f"无效的视频 URL: {url}", code=-1)

    if not url.startswith(('http://', 'https://')):
        logger.error(f"视频下载失败: URL 缺少协议头 '{url}'")
        raise TsingcloudAPIError(f"URL 缺少协议头 (http/https): {url}", code=-1)

    # 用于线程间通信的状态
    download_state = {
        'downloaded': 0,
        'total': 0,
        'finished': False,
        'error': None
    }

    # 心跳线程：每5秒发送一次进度，保持SSE连接
    def heartbeat_thread():
        last_downloaded = -1
        while not download_state['finished']:
            time.sleep(5)
            if download_state['finished']:
                break
            if progress_callback:
                downloaded = download_state['downloaded']
                total = download_state['total']
                # 只在有变化或需要心跳时发送
                if downloaded != last_downloaded or total == 0:
                    progress_callback(downloaded, total)
                    last_downloaded = downloaded

    try:
        # 立即发送开始下载的回调，防止SSE超时
        if progress_callback:
            progress_callback(0, 0)

        # 启动心跳线程
        heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
        heartbeat.start()

        # 设置连接超时(30秒)和读取超时(timeout秒)
        # 云控平台生成视频可能需要较长时间，增加连接超时
        response = requests.get(
            url,
            stream=True,
            timeout=(30, timeout),  # 30秒连接超时（视频生成需要时间）
            verify=False
        )
        response.raise_for_status()

        download_state['total'] = int(response.headers.get('content-length', 0))
        last_callback_time = time.time()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    download_state['downloaded'] += len(chunk)

                    # 每3秒发送一次进度回调
                    if progress_callback and time.time() - last_callback_time >= 3:
                        progress_callback(download_state['downloaded'], download_state['total'])
                        last_callback_time = time.time()

        # 标记下载完成，停止心跳线程
        download_state['finished'] = True

        # 下载完成后发送最终进度回调
        if progress_callback and download_state['total'] > 0:
            progress_callback(download_state['total'], download_state['total'])

        logger.info(f"视频下载完成: {output_path}")
        return True

    except requests.exceptions.Timeout as e:
        download_state['finished'] = True
        logger.error(f"视频下载超时: {e}")
        return False
    except Exception as e:
        download_state['finished'] = True
        logger.error(f"视频下载失败: {e}")
        return False


class DualAccountDownloader:
    """
    双账号视频下载器

    支持两种下载方式：
    1. RTSP方式（wangweiran账号）: 速度快，立即返回RTSP URL
    2. HTTP轮询方式（wangbowen账号）: 需等待服务器生成视频URL

    下载策略：先尝试RTSP，失败后回退到HTTP轮询
    """

    def __init__(
        self,
        # HTTP轮询账号
        http_app_key: str,
        http_password: str,
        # RTSP账号
        rtsp_app_key: str,
        rtsp_password: str,
        # 设备映射文件
        device_mapping_file: str,
        # 通用配置
        base_url: str = "https://rc.ccg.bcavt.com:8760/infraCloud",
        poll_interval: float = 30.0,
        poll_timeout: float = 300.0,
        verify_ssl: bool = False,
        rtsp_download_timeout: int = 420  # RTSP下载超时（秒），默认segment_duration(300)+buffer(120)
    ):
        """
        初始化双账号下载器

        Args:
            http_app_key: HTTP轮询账号
            http_password: HTTP轮询密码
            rtsp_app_key: RTSP账号
            rtsp_password: RTSP密码
            device_mapping_file: 设备映射文件路径
            base_url: API基础URL
            poll_interval: HTTP轮询间隔
            poll_timeout: HTTP轮询超时
            verify_ssl: 是否验证SSL
            rtsp_download_timeout: RTSP下载超时（秒）
        """
        self.rtsp_download_timeout = rtsp_download_timeout

        # HTTP轮询客户端
        self.http_api = TsingcloudAPI(
            app_key=http_app_key,
            password=http_password,
            base_url=base_url,
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
            verify_ssl=verify_ssl
        )

        # RTSP客户端（使用不同账号）
        self.rtsp_api = TsingcloudAPI(
            app_key=rtsp_app_key,
            password=rtsp_password,
            base_url=base_url,
            verify_ssl=verify_ssl
        )

        # 设备映射器
        self.device_mapper = DeviceMapper(device_mapping_file)

        logger.info(f"双账号下载器初始化: HTTP={http_app_key}, RTSP={rtsp_app_key}, RTSP超时={rtsp_download_timeout}秒")

    def download_video(
        self,
        road_id: str,
        start_time: datetime,
        end_time: datetime,
        output_path: str,
        prefer_rtsp: bool = True,
        device_cate: str = "DJ",
        device_index: int = 0,
        channel_num: str = None,
        progress_callback: Callable[[str, int, int], None] = None
    ) -> bool:
        """
        下载视频（先RTSP后轮询）

        Args:
            road_id: 路口ID
            start_time: 开始时间
            end_time: 结束时间
            output_path: 输出文件路径
            prefer_rtsp: 是否优先使用RTSP（默认True）
            device_cate: RTSP设备类型 "DJ"(全景) 或 "KK"(抓拍)
            device_index: RTSP设备索引（同类型中第几个，从0开始）
            channel_num: HTTP轮询时指定的摄像头通道号（可选）
            progress_callback: 进度回调 (method, downloaded, total)

        Returns:
            是否成功
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ===== 方式1: RTSP下载 =====
        if prefer_rtsp:
            device_id = self.device_mapper.get_device_id(road_id, device_cate, device_index)
            if device_id:
                logger.info(f"尝试RTSP下载: road_id={road_id}, device_id={device_id}, cate={device_cate}, index={device_index}")
                if progress_callback:
                    progress_callback("rtsp", 0, 0)

                rtsp_url = self.rtsp_api.get_rtsp_url(device_id, start_time, end_time)
                if rtsp_url:
                    success = download_video_rtsp(
                        rtsp_url, output_path,
                        timeout=self.rtsp_download_timeout,
                        progress_callback=lambda d, t: progress_callback("rtsp", d, t) if progress_callback else None
                    )
                    if success:
                        logger.info(f"RTSP下载成功: {output_path}")
                        return True
                    else:
                        logger.warning("RTSP下载失败，回退到HTTP轮询")
                else:
                    logger.warning("RTSP URL获取失败，回退到HTTP轮询")
            else:
                logger.warning(f"路口{road_id}无{device_cate}设备映射，使用HTTP轮询")

        # ===== 方式2: HTTP轮询下载 =====
        logger.info(f"使用HTTP轮询下载: road_id={road_id}, channel_num={channel_num}")
        if progress_callback:
            progress_callback("http", 0, 0)

        try:
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_str = end_time.strftime("%Y%m%d%H%M%S")

            # 获取摄像头列表
            cameras = self.http_api.get_road_cameras(road_id, start_str, end_str)
            if not cameras:
                logger.error(f"路口{road_id}无摄像头")
                return False

            # 选择摄像头
            target_camera = None

            # 1. 优先使用指定的channel_num
            if channel_num:
                for cam in cameras:
                    if cam.channel_num == channel_num:
                        target_camera = cam
                        break
                if not target_camera:
                    logger.warning(f"未找到指定摄像头 {channel_num}，使用默认全景摄像头")

            # 2. 否则选择全景摄像头
            if not target_camera:
                for cam in cameras:
                    if cam.is_panoramic:
                        target_camera = cam
                        break

            # 3. 兜底使用第一个摄像头
            if not target_camera:
                target_camera = cameras[0]

            # 获取视频URL（带轮询）
            def poll_callback(attempt, max_attempts, msg):
                if progress_callback:
                    progress_callback("http_poll", attempt, max_attempts)

            video_url = self.http_api.get_video_url(
                road_id, target_camera.request_id,
                progress_callback=poll_callback
            )

            # 下载视频
            success = download_video(
                video_url, output_path,
                progress_callback=lambda d, t: progress_callback("http", d, t) if progress_callback else None
            )

            if success:
                logger.info(f"HTTP下载成功: {output_path}")
            return success

        except TsingcloudAPIError as e:
            logger.error(f"HTTP下载失败: {e}")
            return False
        except Exception as e:
            logger.error(f"下载异常: {e}")
            return False

    def get_rtsp_url(
        self,
        road_id: str,
        start_time: datetime,
        end_time: datetime,
        device_cate: str = "DJ",
        device_index: int = 0
    ) -> Optional[str]:
        """
        获取RTSP URL

        Args:
            road_id: 路口ID
            start_time: 开始时间
            end_time: 结束时间
            device_cate: 设备类型 "DJ"(全景) 或 "KK"(抓拍)
            device_index: 设备索引
        """
        device_id = self.device_mapper.get_device_id(road_id, device_cate, device_index)
        if not device_id:
            return None
        return self.rtsp_api.get_rtsp_url(device_id, start_time, end_time)

    def get_road_devices(self, road_id: str, device_cate: str = None) -> List[Dict[str, str]]:
        """
        获取路口的所有可用设备（用于RTSP下载）

        Args:
            road_id: 路口ID
            device_cate: 可选，筛选设备类型 "DJ" 或 "KK"

        Returns:
            设备列表 [{deviceId, sn, cate}, ...]
        """
        return self.device_mapper.get_all_devices(road_id, device_cate)

    def get_http_video_url(
        self,
        road_id: str,
        start_time: datetime,
        end_time: datetime,
        progress_callback: Callable[[int, int, str], None] = None
    ) -> Optional[str]:
        """获取HTTP视频URL（需轮询）"""
        try:
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_str = end_time.strftime("%Y%m%d%H%M%S")

            cameras = self.http_api.get_road_cameras(road_id, start_str, end_str)
            if not cameras:
                return None

            target = next((c for c in cameras if c.is_panoramic), cameras[0])
            return self.http_api.get_video_url(road_id, target.request_id, progress_callback)
        except:
            return None
