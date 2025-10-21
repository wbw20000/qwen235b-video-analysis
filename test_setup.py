"""
测试脚本 - 检查环境配置和依赖
"""

import sys
import subprocess
import io

# 设置标准输出为 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("1. 检查 Python 版本...")
    print("=" * 60)
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 版本过低，需要 Python 3.8 或更高版本")
        return False
    else:
        print("✅ Python 版本符合要求")
        return True

def check_packages():
    """检查必要的包是否已安装"""
    print("\n" + "=" * 60)
    print("2. 检查必要的包...")
    print("=" * 60)

    required_packages = [
        'flask',
        'torch',
        'transformers',
        'PIL',
        'cv2'
    ]

    all_installed = True

    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ Pillow (PIL) 已安装 - 版本: {PIL.__version__}")
            elif package == 'cv2':
                import cv2
                print(f"✅ OpenCV (cv2) 已安装 - 版本: {cv2.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', '未知')
                print(f"✅ {package} 已安装 - 版本: {version}")
        except ImportError:
            print(f"❌ {package} 未安装")
            all_installed = False

    return all_installed

def check_cuda():
    """检查 CUDA 是否可用"""
    print("\n" + "=" * 60)
    print("3. 检查 GPU/CUDA 支持...")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   当前 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")

            # 显示 GPU 显存信息
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i} 显存: {total_memory:.2f} GB")

            return True
        else:
            print("⚠️  CUDA 不可用，将使用 CPU 运行（性能会较慢）")
            return False
    except Exception as e:
        print(f"❌ 检查 CUDA 时出错: {str(e)}")
        return False

def check_directories():
    """检查必要的目录结构"""
    print("\n" + "=" * 60)
    print("4. 检查目录结构...")
    print("=" * 60)

    import os

    directories = ['templates', 'uploads']
    all_exists = True

    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/ 目录存在")
        else:
            print(f"⚠️  {directory}/ 目录不存在，将自动创建")
            os.makedirs(directory, exist_ok=True)

    # 检查 index.html
    if os.path.exists('templates/index.html'):
        print("✅ templates/index.html 文件存在")
    else:
        print("❌ templates/index.html 文件不存在")
        all_exists = False

    return all_exists

def check_model_access():
    """检查是否能访问 Hugging Face"""
    print("\n" + "=" * 60)
    print("5. 检查网络和模型访问...")
    print("=" * 60)

    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=5)
        if response.status_code == 200:
            print("✅ 可以访问 Hugging Face")
            return True
        else:
            print("⚠️  无法访问 Hugging Face，可能需要配置代理")
            return False
    except:
        print("⚠️  无法访问 Hugging Face，可能需要配置代理")
        print("   如果下载模型遇到问题，请尝试:")
        print("   - 配置 HF_ENDPOINT 环境变量")
        print("   - 使用镜像站点")
        return False

def main():
    """主函数"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 10 + "Qwen3-VL 环境检查工具" + " " * 18 + "*")
    print("*" * 60)
    print()

    results = []

    # 执行所有检查
    results.append(("Python 版本", check_python_version()))
    results.append(("依赖包", check_packages()))
    results.append(("CUDA/GPU", check_cuda()))
    results.append(("目录结构", check_directories()))
    results.append(("网络访问", check_model_access()))

    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)

    all_passed = all(result[1] for result in results)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    print("\n" + "=" * 60)

    if all_passed:
        print("🎉 所有检查通过！您可以运行应用了")
        print("\n启动应用:")
        print("  Windows: run.bat")
        print("  Linux/Mac: python app.py")
    else:
        print("⚠️  部分检查未通过，请先解决上述问题")
        print("\n安装依赖:")
        print("  pip install -r requirements.txt")

    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
