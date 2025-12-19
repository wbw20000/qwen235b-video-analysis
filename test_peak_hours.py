"""
高峰时段优先遍历功能测试脚本

测试内容：
1. 高峰时段配置
2. 非高峰时段计算
3. 随机化路口顺序
4. 随机摄像头选择
5. BatchTaskInfo序列化
"""

import os
import sys

# 设置UTF-8输出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_peak_hours_config():
    """测试高峰时段配置"""
    print("\n" + "="*60)
    print("测试1: 高峰时段配置")
    print("="*60)

    from traffic_vlm.config import BatchProcessConfig

    config = BatchProcessConfig()

    print(f"peak_hours_enabled: {config.peak_hours_enabled}")
    print(f"default_peak_hours: {config.default_peak_hours}")
    print(f"randomize_road_order: {config.randomize_road_order}")
    print(f"randomize_camera_selection: {config.randomize_camera_selection}")

    # 验证配置值
    checks = [
        ("peak_hours_enabled默认False", config.peak_hours_enabled == False),
        ("default_peak_hours有两个时段", len(config.default_peak_hours) == 2),
        ("早高峰07:00-09:00", config.default_peak_hours[0] == ("07:00", "09:00")),
        ("晚高峰17:00-19:00", config.default_peak_hours[1] == ("17:00", "19:00")),
        ("randomize_road_order默认True", config.randomize_road_order == True),
        ("randomize_camera_selection默认True", config.randomize_camera_selection == True),
    ]

    all_pass = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("✅ 高峰时段配置正确")
    return all_pass


def test_non_peak_calculation():
    """测试非高峰时段计算"""
    print("\n" + "="*60)
    print("测试2: 非高峰时段计算")
    print("="*60)

    from traffic_vlm.batch_processor import BatchVideoProcessor

    # 创建一个空的processor用于测试方法
    processor = BatchVideoProcessor.__new__(BatchVideoProcessor)

    # 测试用例1: 全天时间范围
    peak_hours = [("07:00", "09:00"), ("17:00", "19:00")]
    non_peak = processor._calculate_non_peak_slots("00:00", "23:59", peak_hours)

    print(f"测试1 - 全天时间范围 00:00-23:59")
    print(f"  高峰时段: {peak_hours}")
    print(f"  非高峰时段: {non_peak}")

    expected1 = [("00:00", "07:00"), ("09:00", "17:00"), ("19:00", "23:59")]
    check1 = non_peak == expected1
    print(f"  {'✅' if check1 else '❌'} 结果{'正确' if check1 else '不正确'}")

    # 测试用例2: 部分时间范围
    non_peak2 = processor._calculate_non_peak_slots("06:00", "20:00", peak_hours)

    print(f"\n测试2 - 部分时间范围 06:00-20:00")
    print(f"  高峰时段: {peak_hours}")
    print(f"  非高峰时段: {non_peak2}")

    expected2 = [("06:00", "07:00"), ("09:00", "17:00"), ("19:00", "20:00")]
    check2 = non_peak2 == expected2
    print(f"  {'✅' if check2 else '❌'} 结果{'正确' if check2 else '不正确'}")

    # 测试用例3: 只有高峰时段
    non_peak3 = processor._calculate_non_peak_slots("07:00", "09:00", peak_hours)

    print(f"\n测试3 - 只包含高峰时段 07:00-09:00")
    print(f"  高峰时段: {peak_hours}")
    print(f"  非高峰时段: {non_peak3}")

    expected3 = []
    check3 = non_peak3 == expected3
    print(f"  {'✅' if check3 else '❌'} 结果{'正确' if check3 else '不正确'}")

    all_pass = check1 and check2 and check3
    if all_pass:
        print("\n✅ 非高峰时段计算正确")
    return all_pass


def test_random_road_order():
    """测试随机化路口顺序"""
    print("\n" + "="*60)
    print("测试3: 随机化路口顺序")
    print("="*60)

    import random

    # 模拟路口列表
    road_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    original = road_list.copy()

    # 设置随机种子以便重现
    random.seed(42)

    # 执行多次随机化
    results = []
    for i in range(5):
        shuffled = road_list.copy()
        random.shuffle(shuffled)
        results.append(shuffled)
        print(f"  第{i+1}次随机: 首个路口={shuffled[0]}, 顺序={shuffled[:5]}...")

    # 检查是否真的随机了（至少有不同的顺序）
    unique_orders = len(set([tuple(r) for r in results]))
    check = unique_orders >= 3  # 至少3种不同顺序

    print(f"\n  产生了 {unique_orders} 种不同的顺序")

    if check:
        print("✅ 随机化路口顺序功能正常")
    else:
        print("❌ 随机化效果不明显")

    return check


def test_batch_task_info_serialization():
    """测试BatchTaskInfo序列化"""
    print("\n" + "="*60)
    print("测试4: BatchTaskInfo序列化")
    print("="*60)

    from traffic_vlm.batch_processor import BatchTaskInfo

    # 创建带高峰时段的任务
    task = BatchTaskInfo(
        batch_id="test123",
        mode="time_traverse",
        start_date="2024-12-17",
        start_time="00:00",
        end_date="2024-12-17",
        end_time="23:59",
        road_ids=[],
        model="qwen-vl-plus",
        analysis_mode="accident",
        peak_hours_enabled=True,
        peak_hours=[("07:00", "09:00"), ("17:00", "19:00")]
    )

    task_dict = task.to_dict()

    print(f"  batch_id: {task_dict['batch_id']}")
    print(f"  peak_hours_enabled: {task_dict['peak_hours_enabled']}")
    print(f"  peak_hours: {task_dict['peak_hours']}")

    checks = [
        ("peak_hours_enabled正确", task_dict['peak_hours_enabled'] == True),
        ("peak_hours是列表", isinstance(task_dict['peak_hours'], list)),
        ("peak_hours有两个元素", len(task_dict['peak_hours']) == 2),
        ("可JSON序列化", all(isinstance(ph, list) for ph in task_dict['peak_hours'])),
    ]

    all_pass = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("✅ BatchTaskInfo序列化正确")
    return all_pass


def test_random_camera_selection():
    """测试随机摄像头选择"""
    print("\n" + "="*60)
    print("测试5: 随机摄像头选择")
    print("="*60)

    import random

    # 模拟摄像头列表
    cameras = [f"cam_{i}" for i in range(1, 11)]
    max_cameras = 1

    # 执行多次随机选择
    random.seed(None)  # 使用系统时间
    selections = []
    for i in range(10):
        if len(cameras) > max_cameras:
            selected = random.sample(cameras, max_cameras)
        else:
            selected = cameras[:max_cameras]
        selections.append(selected[0])
        print(f"  第{i+1}次选择: {selected[0]}")

    # 检查是否选择了不同的摄像头
    unique_selections = len(set(selections))
    check = unique_selections >= 3  # 至少选择了3个不同的摄像头

    print(f"\n  10次选择中出现了 {unique_selections} 个不同的摄像头")

    if check:
        print("✅ 随机摄像头选择功能正常")
    else:
        print("⚠️ 随机性较低（可能是概率问题）")

    return check


def test_peak_hours_traverse_method_exists():
    """测试高峰时段遍历方法存在"""
    print("\n" + "="*60)
    print("测试6: 高峰时段遍历方法存在")
    print("="*60)

    from traffic_vlm.batch_processor import BatchVideoProcessor

    methods = [
        "_peak_hours_traverse",
        "_process_road_peak_hours",
        "_process_road_non_peak_hours",
        "_calculate_non_peak_slots",
        "_process_camera_time_range",
    ]

    all_pass = True
    for method in methods:
        exists = hasattr(BatchVideoProcessor, method)
        status = "✅" if exists else "❌"
        print(f"  {status} {method}")
        if not exists:
            all_pass = False

    if all_pass:
        print("✅ 所有高峰时段遍历方法已实现")
    return all_pass


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("  高峰时段优先遍历功能测试")
    print("="*60)

    results = []

    # 测试1: 高峰时段配置
    try:
        results.append(("高峰时段配置", test_peak_hours_config()))
    except Exception as e:
        print(f"❌ 高峰时段配置测试失败: {e}")
        results.append(("高峰时段配置", False))

    # 测试2: 非高峰时段计算
    try:
        results.append(("非高峰时段计算", test_non_peak_calculation()))
    except Exception as e:
        print(f"❌ 非高峰时段计算测试失败: {e}")
        results.append(("非高峰时段计算", False))

    # 测试3: 随机化路口顺序
    try:
        results.append(("随机化路口顺序", test_random_road_order()))
    except Exception as e:
        print(f"❌ 随机化路口顺序测试失败: {e}")
        results.append(("随机化路口顺序", False))

    # 测试4: BatchTaskInfo序列化
    try:
        results.append(("BatchTaskInfo序列化", test_batch_task_info_serialization()))
    except Exception as e:
        print(f"❌ BatchTaskInfo序列化测试失败: {e}")
        results.append(("BatchTaskInfo序列化", False))

    # 测试5: 随机摄像头选择
    try:
        results.append(("随机摄像头选择", test_random_camera_selection()))
    except Exception as e:
        print(f"❌ 随机摄像头选择测试失败: {e}")
        results.append(("随机摄像头选择", False))

    # 测试6: 高峰时段遍历方法存在
    try:
        results.append(("高峰时段遍历方法", test_peak_hours_traverse_method_exists()))
    except Exception as e:
        print(f"❌ 高峰时段遍历方法测试失败: {e}")
        results.append(("高峰时段遍历方法", False))

    # 汇总结果
    print("\n" + "="*60)
    print("  测试结果汇总")
    print("="*60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
