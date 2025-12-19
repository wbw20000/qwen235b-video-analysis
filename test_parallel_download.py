"""
并行下载能力测试脚本

测试内容：
1. API是否支持并行请求视频URL
2. 视频是否可以并行下载
3. 不同并行度的性能对比
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# 设置UTF-8输出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_vlm.tsingcloud_api import TsingcloudAPI, download_video, TsingcloudAPIError
from traffic_vlm.config import TsingcloudConfig


def test_api_connection(app_key=None, password=None):
    """测试API连接"""
    print("\n" + "="*60)
    print("测试1: API连接")
    print("="*60)

    # 优先使用传入的参数，否则使用环境变量
    if not app_key or not password:
        config = TsingcloudConfig()
        app_key = app_key or config.app_key
        password = password or config.password

    if not app_key or not password:
        print("警告: 未配置TSINGCLOUD_APP_KEY或TSINGCLOUD_PASSWORD环境变量")
        return None

    try:
        api = TsingcloudAPI(
            app_key=app_key,
            password=password,
            request_interval=0.5  # 测试时减少请求间隔
        )
        token = api.get_token()
        print(f"Token获取成功: {token[:20]}...")
        return api
    except Exception as e:
        print(f"API连接失败: {e}")
        return None


def test_parallel_camera_requests(api: TsingcloudAPI, road_ids: list, time_range: tuple):
    """测试并行请求多个路口的摄像头列表"""
    print("\n" + "="*60)
    print("测试2: 并行请求摄像头列表")
    print("="*60)

    start_str, end_str = time_range

    # 串行请求
    print("\n--- 串行请求 ---")
    start_time = time.time()
    serial_results = []
    for road_id in road_ids:
        try:
            cameras = api.get_road_cameras(road_id, start_str, end_str)
            serial_results.append((road_id, len(cameras), None))
            print(f"  路口 {road_id}: {len(cameras)} 个摄像头")
        except Exception as e:
            serial_results.append((road_id, 0, str(e)))
            print(f"  路口 {road_id}: 错误 - {e}")
    serial_time = time.time() - start_time
    print(f"串行总耗时: {serial_time:.2f}秒")

    # 并行请求
    print("\n--- 并行请求 ---")
    start_time = time.time()
    parallel_results = []

    def fetch_cameras(road_id):
        try:
            cameras = api.get_road_cameras(road_id, start_str, end_str)
            return (road_id, len(cameras), None)
        except Exception as e:
            return (road_id, 0, str(e))

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_cameras, rid): rid for rid in road_ids}
        for future in as_completed(futures):
            result = future.result()
            parallel_results.append(result)
            road_id, count, error = result
            if error:
                print(f"  路口 {road_id}: 错误 - {error}")
            else:
                print(f"  路口 {road_id}: {count} 个摄像头")

    parallel_time = time.time() - start_time
    print(f"并行总耗时: {parallel_time:.2f}秒")

    # 分析结果
    print("\n--- 分析 ---")
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    print(f"加速比: {speedup:.2f}x")

    # 检查是否有API限流错误
    errors = [r for r in parallel_results if r[2]]
    if errors:
        print(f"警告: {len(errors)} 个请求出错")
        for road_id, _, error in errors:
            if "限流" in str(error) or "rate" in str(error).lower():
                print(f"  可能是API限流: {error}")

    return {
        "serial_time": serial_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "errors": len(errors)
    }


def test_parallel_video_url_requests(api: TsingcloudAPI, cameras_info: list):
    """测试并行请求视频URL"""
    print("\n" + "="*60)
    print("测试3: 并行请求视频URL")
    print("="*60)

    if not cameras_info:
        print("跳过: 无摄像头信息")
        return None

    # 只测试前3个摄像头
    test_cameras = cameras_info[:3]
    print(f"测试 {len(test_cameras)} 个摄像头")

    results = []

    def fetch_video_url(camera_info):
        road_id, camera = camera_info
        start_time = time.time()
        try:
            url = api.get_video_url(road_id, camera.request_id)
            elapsed = time.time() - start_time
            return {
                "road_id": road_id,
                "channel_num": camera.channel_num,
                "success": True,
                "url": url[:50] if url else None,
                "time": elapsed
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "road_id": road_id,
                "channel_num": camera.channel_num,
                "success": False,
                "error": str(e),
                "time": elapsed
            }

    # 并行请求
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_video_url, cam) for cam in test_cameras]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"  {result['channel_num']}: 成功 ({result['time']:.1f}秒)")
            else:
                print(f"  {result['channel_num']}: 失败 - {result['error']}")

    total_time = time.time() - start_time
    print(f"并行总耗时: {total_time:.2f}秒")

    successful = [r for r in results if r["success"]]
    print(f"成功率: {len(successful)}/{len(results)}")

    return {
        "total_time": total_time,
        "success_count": len(successful),
        "total_count": len(results),
        "results": results
    }


def test_parallel_downloads(urls_info: list, output_dir: str):
    """测试并行下载视频"""
    print("\n" + "="*60)
    print("测试4: 并行下载视频")
    print("="*60)

    if not urls_info:
        print("跳过: 无可用URL")
        return None

    os.makedirs(output_dir, exist_ok=True)

    def download_single(info):
        url = info["url"]
        channel = info["channel_num"]
        output_path = os.path.join(output_dir, f"test_{channel}.mp4")

        start_time = time.time()
        try:
            success = download_video(url, output_path, timeout=120)
            elapsed = time.time() - start_time
            file_size = os.path.getsize(output_path) if success else 0
            return {
                "channel": channel,
                "success": success,
                "time": elapsed,
                "size_mb": file_size / 1024 / 1024
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "channel": channel,
                "success": False,
                "error": str(e),
                "time": elapsed
            }

    # 并行下载（2路）
    print("\n--- 2路并行下载 ---")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download_single, info) for info in urls_info[:2]]
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                print(f"  {result['channel']}: {result['size_mb']:.1f}MB ({result['time']:.1f}秒)")
            else:
                print(f"  {result['channel']}: 失败 - {result.get('error', '未知')}")

    total_time_2 = time.time() - start_time
    print(f"2路并行总耗时: {total_time_2:.2f}秒")

    return {"parallel_2_time": total_time_2}


def test_api_rate_limit(api: TsingcloudAPI, road_id: str, time_range: tuple):
    """测试API限流阈值"""
    print("\n" + "="*60)
    print("测试5: API限流阈值探测")
    print("="*60)

    start_str, end_str = time_range

    # 快速连续请求测试
    request_counts = [5, 10, 15]

    for count in request_counts:
        print(f"\n--- 连续 {count} 次请求 ---")
        successes = 0
        errors = 0
        start_time = time.time()

        for i in range(count):
            try:
                # 临时禁用请求间隔
                old_interval = api.request_interval
                api.request_interval = 0.1
                cameras = api.get_road_cameras(road_id, start_str, end_str)
                api.request_interval = old_interval
                successes += 1
            except Exception as e:
                errors += 1
                error_str = str(e).lower()
                if "限流" in error_str or "rate" in error_str or "429" in error_str:
                    print(f"  请求 {i+1}: 触发限流 - {e}")
                    break
                else:
                    print(f"  请求 {i+1}: 其他错误 - {e}")

        elapsed = time.time() - start_time
        print(f"结果: 成功 {successes}/{count}, 耗时 {elapsed:.2f}秒")

        if errors > 0:
            print(f"建议: 请求间隔应 >= {elapsed/count:.2f}秒")
            break

        time.sleep(2)  # 冷却

    return {"max_continuous": successes}


def main():
    """运行所有测试"""
    import argparse
    parser = argparse.ArgumentParser(description='云控智行API并行能力测试')
    parser.add_argument('--app-key', default='wangbowen', help='API App Key')
    parser.add_argument('--password', default='YwKSBcgWUI6', help='API Password')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  云控智行API并行能力测试")
    print("="*60)

    # 测试配置
    test_road_ids = ["1", "2", "3"]  # 测试用路口ID

    # 使用较短的时间范围（5分钟）
    now = datetime.now()
    # 使用昨天的时间范围（确保有历史视频）
    yesterday = now - timedelta(days=1)
    start_dt = yesterday.replace(hour=8, minute=0, second=0)
    end_dt = start_dt + timedelta(minutes=5)

    time_range = (
        start_dt.strftime("%Y%m%d%H%M%S"),
        end_dt.strftime("%Y%m%d%H%M%S")
    )
    print(f"\n测试时间范围: {start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')}")

    # 测试1: API连接
    api = test_api_connection(args.app_key, args.password)
    if not api:
        print("\nAPI连接失败，终止测试")
        return

    # 测试2: 并行请求摄像头列表
    camera_test = test_parallel_camera_requests(api, test_road_ids, time_range)

    # 收集摄像头信息用于后续测试
    cameras_info = []
    for road_id in test_road_ids:
        try:
            cameras = api.get_road_cameras(road_id, time_range[0], time_range[1])
            panoramic = [c for c in cameras if c.is_panoramic]
            if panoramic:
                cameras_info.append((road_id, panoramic[0]))
        except:
            pass

    # 测试3: 并行请求视频URL
    url_test = test_parallel_video_url_requests(api, cameras_info)

    # 测试4: 并行下载（如果有URL）
    if url_test and url_test.get("results"):
        successful_urls = [
            {"url": r["url"], "channel_num": r["channel_num"]}
            for r in url_test["results"] if r["success"]
        ]
        if successful_urls:
            output_dir = "temp/parallel_test"
            download_test = test_parallel_downloads(successful_urls[:2], output_dir)

    # 测试5: API限流阈值
    rate_limit_test = test_api_rate_limit(api, test_road_ids[0], time_range)

    # 汇总
    print("\n" + "="*60)
    print("  测试结果汇总")
    print("="*60)

    print("\n1. API摄像头列表请求:")
    if camera_test:
        print(f"   - 串行 vs 并行加速比: {camera_test['speedup']:.2f}x")
        print(f"   - 并行错误数: {camera_test['errors']}")
        if camera_test['speedup'] > 1.5:
            print("   - 结论: API支持并行请求")
        else:
            print("   - 结论: API可能有限流，建议串行")

    print("\n2. 视频URL请求:")
    if url_test:
        print(f"   - 成功率: {url_test['success_count']}/{url_test['total_count']}")
        print(f"   - 总耗时: {url_test['total_time']:.1f}秒")

    print("\n3. 限流阈值:")
    if rate_limit_test:
        print(f"   - 连续请求最大成功数: {rate_limit_test['max_continuous']}")

    print("\n建议配置:")
    print("   - concurrent_cameras: 1-2（视API限流情况）")
    print("   - request_interval: 1.0秒（保守）或 0.5秒（激进）")


if __name__ == "__main__":
    main()
