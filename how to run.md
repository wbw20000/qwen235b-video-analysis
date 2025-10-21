# Qwen3-VL 视频分析项目 - 启动指南

## 项目简介

这是一个基于阿里云 DashScope API 的 Qwen3-VL 视频分析 Web 应用，可以上传视频并使用 AI 模型进行智能分析。

## 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高版本
- **FFmpeg**: 用于视频压缩（可选，但强烈推荐）

## 安装步骤

### 1. 安装 Python 依赖

项目使用虚拟环境管理依赖，确保已激活虚拟环境：

```bash
# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

安装所需的 Python 包：

```bash
pip install flask openai
```

或使用 requirements.txt（如果有）：

```bash
pip install -r requirements.txt
```

### 2. 安装 FFmpeg（可选但推荐）

FFmpeg 用于自动压缩大视频文件，使其符合 API 限制（10MB）。

#### Windows:

1. 下载 FFmpeg: https://ffmpeg.org/download.html
2. 推荐使用 gyan.dev 构建版本: https://www.gyan.dev/ffmpeg/builds/
3. 下载 "ffmpeg-release-essentials.zip"
4. 解压到 `C:\ffmpeg`
5. 添加到系统 PATH: `C:\ffmpeg\bin`

验证安装：

```bash
ffmpeg -version
```

#### Linux:

```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS:

```bash
brew install ffmpeg
```

### 3. 配置 API Key

项目需要阿里云 DashScope API Key 才能使用 Qwen3-VL 模型。

#### 获取 API Key:

访问: https://dashscope.console.aliyun.com/apiKey

#### 配置方式 1: 环境变量（推荐）

**Windows:**
```bash
set DASHSCOPE_API_KEY=your_api_key_here
```

**Linux/macOS:**
```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

#### 配置方式 2: 直接修改代码

编辑 `app.py` 文件，找到第 36 行：

```python
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-your-api-key-here')
```

将 `'sk-your-api-key-here'` 替换为你的实际 API Key。

## 启动项目

### 方法 1: 使用 Python 直接启动

```bash
# Windows (使用虚拟环境)
.\venv\Scripts\python app.py

# Linux/macOS
python app.py
```

### 方法 2: 使用启动脚本

```bash
# Windows
start.bat

# 或
run.bat
```

### 启动成功

当看到以下输出时，说明服务启动成功：

```
============================================================
Qwen3-VL 视频分析服务 (DashScope API)
============================================================

✓ DashScope API Key 已配置
  API Key: sk-51756...

可用模型: 3 个
  - qwen-vl-max: Qwen-VL-Max（最强视觉理解能力）
  - qwen-vl-plus: Qwen-VL-Plus（推荐，性价比高）
  - qwen3-vl-plus: Qwen3-VL-Plus（最新版本）

============================================================
服务启动在: http://localhost:5000
============================================================
```

## 访问应用

在浏览器中打开以下任一地址：

- http://localhost:5000
- http://127.0.0.1:5000

## 使用说明

### 上传视频

1. 点击"选择视频文件"按钮
2. 选择支持的视频格式: MP4, AVI, MKV, MOV, FLV, WMV
3. 最大文件大小: 500MB

### 选择模型

- **qwen-vl-max**: 最强视觉理解能力（推荐用于复杂场景）
- **qwen-vl-plus**: 性价比高（推荐日常使用）
- **qwen3-vl-plus**: 最新版本

### 输入提问

在提问框中输入你想问的问题，例如：

- "请详细描述视频中发生了什么"
- "这个视频的主要内容是什么？"
- "视频中有哪些物体？"
- "分析这个视频的场景和活动"

### 自动压缩

- 如果视频大于 7MB，系统会自动使用 FFmpeg 压缩到约 6.5MB
- 可以在设置中关闭自动压缩功能
- 压缩后的视频可以下载

### 查看结果

点击"分析视频"后：

1. 系统会上传并处理视频
2. 如果需要，会自动压缩视频
3. 调用 AI 模型进行分析
4. 显示分析结果

## 常见问题

### Q: 提示"未安装 FFmpeg"

**A:** 请按照上述步骤安装 FFmpeg 并添加到系统 PATH。安装后重启应用。

### Q: 视频太大无法分析

**A:**
1. 确保已安装 FFmpeg 并启用自动压缩
2. 或手动压缩视频到 7MB 以下
3. API 限制: base64 编码后不能超过 10MB

### Q: API 调用失败

**A:**
1. 检查 API Key 是否正确配置
2. 确认 API Key 有效且有足够额度
3. 检查网络连接

### Q: 分析速度慢

**A:**
1. 大视频需要较长编码时间
2. API 调用需要网络传输时间
3. 建议使用较小的视频或启用压缩

## API 端点

项目提供以下 API 端点：

- `GET /` - 主页
- `POST /analyze` - 分析视频
- `GET /download/<filename>` - 下载压缩视频
- `GET /health` - 健康检查
- `GET /config` - 获取配置信息

## 技术栈

- **后端**: Flask (Python Web 框架)
- **AI 模型**: 阿里云 DashScope API (Qwen3-VL)
- **视频处理**: FFmpeg
- **前端**: HTML + JavaScript + Bootstrap

## 项目结构

```
qwen235b/
├── app.py                 # 主应用程序
├── templates/             # HTML 模板
│   └── index.html        # 前端界面
├── uploads/              # 上传的视频文件
├── venv/                 # Python 虚拟环境
├── requirements.txt      # Python 依赖
├── how to run.md         # 本文档
├── README.md             # 项目说明
├── CLAUDE.md             # 项目目标
└── QUICKSTART.md         # 快速开始

```

## 停止服务

在运行 Flask 的终端窗口按 `Ctrl + C` 即可停止服务。

## 注意事项

1. **开发模式**: 当前使用 Flask 开发服务器，仅适合开发和测试
2. **生产环境**: 如需部署到生产环境，请使用 Gunicorn 或 uWSGI
3. **API 限制**: 注意 DashScope API 的调用额度和频率限制
4. **文件清理**: 定期清理 `uploads` 文件夹中的临时文件

## 联系支持

- DashScope API 文档: https://help.aliyun.com/zh/dashscope/
- FFmpeg 官网: https://ffmpeg.org/
- 项目 GitHub: （如有）

---

**祝使用愉快！** 🚀
