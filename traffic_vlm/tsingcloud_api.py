"""
云控智行 API 客户端

提供历史视频查询与下载功能：
1. Token认证（SHA256签名，7天有效期）
2. 获取路口摄像头列表
3. 获取视频下载URL（支持轮询等待）
"""
from __future__ import annotations

import hashlib
import time
import logging
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable

# 禁用SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


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

    # 视频生成状态码
    STATUS_GENERATING = 30056  # URL生成中
    STATUS_OK = 0              # 成功

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

            if status == self.STATUS_OK and data:
                # 成功获取URL
                video_url = data if isinstance(data, str) else data.get("url", "")
                # 校验 URL 有效性（API 可能返回 "null" 字符串）
                if video_url and video_url.lower() not in ('null', 'none', ''):
                    if video_url.startswith(('http://', 'https://')):
                        logger.info(f"视频URL获取成功: {video_url[:50]}...")
                        return video_url
                    else:
                        logger.warning(f"视频URL格式异常: {video_url}")
                raise TsingcloudAPIError(f"返回数据中无有效URL (data={data})", code=status, response=result)

            elif status == self.STATUS_GENERATING:
                # 视频生成中，分段等待并发送心跳
                logger.debug(f"视频生成中，{self.poll_interval}秒后重试 ({attempt+1}/{max_attempts})")
                # 分段睡眠，每5秒发送一次心跳回调
                heartbeat_interval = 5
                elapsed = 0
                while elapsed < self.poll_interval:
                    time.sleep(heartbeat_interval)
                    elapsed += heartbeat_interval
                    if progress_callback:
                        progress_callback(attempt + 1, max_attempts, f"等待视频生成中... ({elapsed}s/{int(self.poll_interval)}s)")

            else:
                # 其他错误
                raise TsingcloudAPIError(
                    f"获取视频URL失败: {msg}",
                    code=status,
                    response=result
                )

        raise TsingcloudAPIError(
            f"获取视频URL超时（等待{self.poll_timeout}秒）",
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
