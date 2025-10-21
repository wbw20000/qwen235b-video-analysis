# Qwen3-VL 视频分析 Web 应用

通过阿里云 DashScope API 调用 Qwen3-VL 模型进行视频分析，无需本地部署大模型。

## 功能特性

- 🎬 **视频上传**: 支持拖拽上传或点击选择视频文件
- 🤖 **智能分析**: 使用阿里云 DashScope API 调用 Qwen3-VL 模型
- 💬 **自定义提问**: 可以针对视频内容提出特定问题
- 🎯 **多模型选择**: 支持 Qwen-VL-Plus、Qwen3-VL-Plus、Qwen-VL-Max
- 🎨 **美观界面**: 现代化的 Web 界面，操作简单直观
- ☁️ **云端部署**: 无需GPU，无需下载大模型，开箱即用

## 支持的视频格式

- MP4
- AVI
- MKV
- MOV
- FLV
- WMV

## 系统要求

### 硬件要求

- ✅ **无GPU要求** - 使用云端API，任何电脑都可以运行
- **内存**: 至少 2GB RAM
- **网络**: 稳定的互联网连接

### 软件要求

- Python 3.8+
- Windows / Linux / macOS

## 安装步骤

### 1. 克隆或下载项目

```bash
cd C:\project2025\qwen235b
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项非常简单：
- `flask` - Web 框架
- `openai` - OpenAI 兼容的客户端库

### 3. 配置 DashScope API Key

#### 获取 API Key

1. 访问阿里云 DashScope 控制台：https://dashscope.console.aliyun.com/apiKey
2. 登录您的阿里云账号
3. 创建或复制您的 API Key

#### 配置 API Key

**方法1：环境变量（推荐）**

Windows:
```bash
set DASHSCOPE_API_KEY=sk-your-api-key-here
```

Linux/Mac:
```bash
export DASHSCOPE_API_KEY=sk-your-api-key-here
```

**方法2：直接在代码中配置**

编辑 `app.py`，在第26行左右找到：
```python
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
```

改为：
```python
DASHSCOPE_API_KEY = 'sk-your-api-key-here'
```

**方法3：使用 .env 文件**

```bash
# 复制示例文件
copy .env.example .env

# 编辑 .env 文件，填入你的 API Key
DASHSCOPE_API_KEY=sk-your-api-key-here
```

然后在 `app.py` 中添加：
```python
from dotenv import load_dotenv
load_dotenv()
```

## 使用方法

### 启动应用

```bash
python app.py
```

启动后，程序会在 `http://localhost:5000` 运行。

### 使用 Web 界面

1. 打开浏览器，访问 `http://localhost:5000`
2. 选择模型（推荐使用 Qwen-VL-Plus）
3. 上传视频文件（拖拽或点击选择）
4. 输入您的问题（可选）
5. 点击"开始分析"按钮
6. 等待分析完成，查看结果

### 可用模型

| 模型 | 描述 | 适用场景 |
|------|------|----------|
| **qwen-vl-plus** | 推荐，性价比高 | 日常使用，路测视频分析 |
| **qwen3-vl-plus** | 最新版本 | 需要最新特性时使用 |
| **qwen-vl-max** | 最强视觉理解能力 | 复杂场景，高精度要求 |

## 示例问题

针对路测视频，您可以提问：

- "请详细描述这个视频中发生了什么。"
- "视频中出现了哪些交通标志？"
- "车辆在视频中的行驶速度如何？"
- "有哪些潜在的安全隐患？"
- "请识别视频中的道路类型和环境。"
- "视频中有哪些其他车辆和行人？"

## 项目结构

```
qwen235b/
├── app.py                 # Flask 后端主程序
├── templates/
│   └── index.html        # 前端页面
├── uploads/              # 上传的视频存储目录（自动创建）
├── requirements.txt      # Python 依赖
├── .env.example          # 环境变量示例
├── README.md            # 项目文档
├── QUICKSTART.md        # 快速开始指南
└── CLAUDE.md            # 项目目标和方法
```

## API 接口

### POST /analyze

分析视频内容

**参数**:
- `video` (file): 视频文件
- `prompt` (string): 用户问题
- `model` (string): 模型选择（qwen-vl-plus/qwen3-vl-plus/qwen-vl-max）

**响应**:
```json
{
  "success": true,
  "result": "视频分析结果...",
  "video_name": "video.mp4",
  "model": "qwen-vl-plus"
}
```

### GET /health

健康检查

**响应**:
```json
{
  "status": "ok",
  "api_configured": true,
  "available_models": ["qwen-vl-plus", "qwen3-vl-plus", "qwen-vl-max"]
}
```

### GET /config

获取配置信息

**响应**:
```json
{
  "api_key_configured": true,
  "available_models": {...},
  "max_video_size_mb": 500,
  "allowed_extensions": ["mp4", "avi", "mkv", "mov", "flv", "wmv"]
}
```

## 常见问题

### Q: 如何获取 DashScope API Key？
A: 访问 https://dashscope.console.aliyun.com/apiKey 登录阿里云账号即可获取。

### Q: API 如何收费？
A: DashScope API 按调用量计费。具体价格请参考：https://help.aliyun.com/zh/model-studio/developer-reference/billing

### Q: 视频分析需要多长时间？
A: 通常几秒到几十秒，取决于视频长度和网络速度。

### Q: 支持多长的视频？
A: 推荐 2 秒到 20 分钟的视频，文件大小不超过 500MB。

### Q: 没有阿里云账号怎么办？
A: 需要注册阿里云账号并开通 DashScope 服务。新用户通常有免费额度。

### Q: API Key 安全吗？
A: 请不要将 API Key 提交到公共代码仓库。使用环境变量或 .env 文件配置。

## 技术栈

- **后端**: Flask (Python)
- **前端**: HTML5 + CSS3 + JavaScript
- **AI 服务**: 阿里云 DashScope API
- **模型**: Qwen3-VL 系列

## 优势对比

### 使用 DashScope API（本项目）

✅ 无需下载大模型（节省数百GB空间）
✅ 无需GPU（任何电脑都能运行）
✅ 开箱即用（安装依赖即可使用）
✅ 按需付费（用多少付多少）
✅ 自动更新（始终使用最新模型）
✅ 稳定可靠（云端部署，高可用）

### 本地部署模型

❌ 需要下载 200GB+ 模型
❌ 需要 40GB+ 显存的GPU
❌ 部署复杂，环境配置困难
❌ 固定成本高
❌ 需要手动更新模型
❌ 硬件故障风险

## 参考资料

- [DashScope API 文档](https://help.aliyun.com/zh/dashscope/)
- [Qwen-VL 模型文档](https://help.aliyun.com/zh/model-studio/vision)
- [视频理解功能说明](https://help.aliyun.com/zh/model-studio/qwenvl-video-understanding)
- [API Key 获取](https://dashscope.console.aliyun.com/apiKey)
- [计费说明](https://help.aliyun.com/zh/model-studio/developer-reference/billing)

## 许可证

本项目遵循 MIT 许可证。使用 DashScope API 需遵守阿里云服务协议。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 作者

基于 CLAUDE.md 文档要求开发

---

**注意**: 本项目使用阿里云 DashScope API，需要网络连接和有效的 API Key。API 调用会产生费用，请注意控制使用量。
