# 快速开始指南 - Qwen3-VL 视频分析项目

## 📋 前提条件

- Python 3.8+ 已安装
- 已有阿里云 DashScope API Key（如没有，需先注册获取）
- Windows 10/11 或 Linux/macOS 系统

---

## 🚀 完整启动步骤（从零开始）

### 第 1 步：打开项目目录（10秒）

**Windows:**

1. 按 `Win + R` 打开运行窗口
2. 输入 `cmd` 回车，打开命令提示符
3. 进入项目目录：

```bash
cd C:\project2025\qwen235b
```

**或者:** 在项目文件夹中，按住 `Shift` 键，右键点击空白处，选择"在此处打开命令窗口"或"在终端中打开"

**Linux/macOS:**

```bash
cd /path/to/qwen235b
```

---

### 第 2 步：激活虚拟环境（10秒）

**Windows:**

```bash
.\venv\Scripts\activate
```

**成功标志:** 命令提示符前面会出现 `(venv)` 标记，例如：
```
(venv) C:\project2025\qwen235b>
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

**成功标志:** 终端前面会出现 `(venv)` 标记

**注意:** 如果虚拟环境不存在，需要先创建：
```bash
# Windows
python -m venv venv

# Linux/macOS
python3 -m venv venv
```

---

### 第 3 步：安装 Python 依赖包（30秒）

确保虚拟环境已激活，然后安装必需的包：

**Windows:**

```bash
.\venv\Scripts\pip install flask openai
```

**Linux/macOS:**

```bash
pip install flask openai
```

**安装成功标志:**
```
Successfully installed flask-3.x.x openai-1.x.x ...
```

**如果有 requirements.txt 文件:**

```bash
pip install -r requirements.txt
```

---

### 第 4 步：安装 FFmpeg（可选但强烈推荐，3-5分钟）

FFmpeg 用于自动压缩大视频文件（>7MB），使其符合 API 的 10MB 限制。

#### Windows 详细安装步骤：

**方法 1: 手动安装（推荐）**

1. **下载 FFmpeg:**
   - 访问: https://www.gyan.dev/ffmpeg/builds/
   - 下载 "ffmpeg-release-essentials.zip"（约 80MB）

2. **解压文件:**
   - 解压到 `C:\ffmpeg` 目录
   - 完整路径应该是: `C:\ffmpeg\bin\ffmpeg.exe`

3. **添加到系统 PATH:**

   a. 按 `Win + X`，选择"系统"

   b. 点击"高级系统设置"

   c. 点击"环境变量"

   d. 在"系统变量"中找到 `Path`，双击编辑

   e. 点击"新建"，添加: `C:\ffmpeg\bin`

   f. 点击"确定"保存所有窗口

   g. **重新打开命令提示符**（必须重启终端才能生效）

4. **验证安装:**

```bash
ffmpeg -version
```

**成功标志:** 显示 FFmpeg 版本信息
```
ffmpeg version 8.0-essentials_build-www.gyan.dev ...
```

**方法 2: 使用 Chocolatey（如果已安装）**

```bash
choco install ffmpeg
```

#### Linux 安装：

```bash
sudo apt update
sudo apt install ffmpeg

# 验证
ffmpeg -version
```

#### macOS 安装：

```bash
brew install ffmpeg

# 验证
ffmpeg -version
```

**详细安装教程:** 查看项目中的 `FFMPEG_INSTALL.md` 文件

**如果跳过此步骤:** 只能分析小于 7MB 的视频文件

---

### 第 5 步：配置 API Key（1-2分钟）

#### 5.1 获取 API Key

1. 访问阿里云 DashScope 控制台: https://dashscope.console.aliyun.com/apiKey
2. 登录阿里云账号（没有账号需要先注册）
3. 创建或查看 API Key
4. 复制你的 API Key（格式: `sk-xxxxxxxxxxxxxxxx`）

#### 5.2 配置 API Key

**方式 1: 使用环境变量（推荐，临时有效）**

**Windows:**

```bash
set DASHSCOPE_API_KEY=sk-你的API密钥
```

例如：
```bash
set DASHSCOPE_API_KEY=sk-5175677ff9b4459aa45ce7ec28037515
```

**Linux/macOS:**

```bash
export DASHSCOPE_API_KEY=sk-你的API密钥
```

**注意:** 这种方式在关闭命令窗口后失效，下次需要重新设置

**方式 2: 直接修改代码（永久有效）**

1. 使用文本编辑器（如 Notepad++、VS Code）打开 `app.py`
2. 找到第 36 行：
   ```python
   DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-5175677ff9b4459aa45ce7ec28037515')
   ```
3. 将引号内的密钥替换为你的 API Key
4. 保存文件

---

### 第 6 步：启动项目（10秒）

确保：
- ✅ 虚拟环境已激活（提示符前有 `(venv)` 标记）
- ✅ 在项目目录 `C:\project2025\qwen235b`
- ✅ API Key 已配置

**启动命令:**

**Windows:**

```bash
.\venv\Scripts\python app.py
```

**Linux/macOS:**

```bash
python app.py
```

**启动成功标志 - 你应该看到以下输出:**

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

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://172.18.0.1:5000
Press CTRL+C to quit
```

**如果看到 "未配置 API Key" 警告:**
- 返回第 5 步重新配置 API Key

**如果看到 "未安装 FFmpeg" 提示:**
- 这是警告信息，不影响启动
- 但只能处理小于 7MB 的视频
- 建议按第 4 步安装 FFmpeg

---

### 第 7 步：访问 Web 界面（5秒）

1. **打开浏览器**（Chrome、Edge、Firefox 等）

2. **访问以下任一地址:**
   - http://localhost:5000
   - http://127.0.0.1:5000

3. **看到 Web 界面** - 成功！🎉

---

### 第 8 步：测试视频分析

1. **准备测试视频:**
   - 找一个小于 50MB 的视频文件
   - 支持格式: MP4, AVI, MKV, MOV, FLV, WMV
   - 推荐使用 MP4 格式

2. **上传视频:**
   - 点击"选择视频文件"按钮
   - 或直接拖拽视频到上传区域

3. **选择模型:**
   - 推荐: **qwen-vl-plus**（性价比高）
   - 复杂场景: qwen-vl-max
   - 最新功能: qwen3-vl-plus

4. **输入提问:**

   示例问题：
   ```
   请详细描述这个视频中发生了什么
   ```

   或路测视频专用问题：
   ```
   请分析视频中的道路情况、交通标志和车辆行为
   ```

5. **启用自动压缩:**
   - 勾选"自动压缩大视频"（如果已安装 FFmpeg）
   - 大于 7MB 的视频会自动压缩到约 6.5MB

6. **点击"分析视频"按钮**

7. **等待结果:**
   - 上传视频: 几秒
   - 压缩视频（如需要）: 10-30秒
   - AI 分析: 5-15秒
   - 总计通常 < 1 分钟

8. **查看分析结果** - 显示在页面下方

---

## 🎯 完整操作示例（Windows）

以下是一个完整的启动流程示例：

```bash
# 1. 打开命令提示符，进入项目目录
cd C:\project2025\qwen235b

# 2. 激活虚拟环境
.\venv\Scripts\activate

# 3. 确认依赖已安装（如果首次运行）
.\venv\Scripts\pip install flask openai

# 4. 验证 FFmpeg（可选）
ffmpeg -version

# 5. 设置 API Key
set DASHSCOPE_API_KEY=sk-你的API密钥

# 6. 启动项目
.\venv\Scripts\python app.py

# 7. 浏览器访问 http://localhost:5000

# 8. 停止服务: 按 Ctrl+C
```

---

## 🔴 常见启动问题排查

### 问题 1: 命令提示符中输入 `.\venv\Scripts\activate` 后没有显示 `(venv)`

**原因:** 虚拟环境不存在或路径错误

**解决:**
```bash
# 创建虚拟环境
python -m venv venv

# 再次激活
.\venv\Scripts\activate
```

### 问题 2: 提示 "flask: command not found" 或 "No module named 'flask'"

**原因:** 依赖包未安装或虚拟环境未激活

**解决:**
```bash
# 确保虚拟环境已激活（命令前有 (venv) 标记）
.\venv\Scripts\pip install flask openai
```

### 问题 3: 启动后显示 "未配置 API Key"

**原因:** API Key 未设置或设置错误

**解决:**
```bash
# 重新设置环境变量
set DASHSCOPE_API_KEY=sk-你的真实API密钥

# 或直接修改 app.py 第 36 行
```

### 问题 4: 网页提示 "未安装 FFmpeg"

**原因:** FFmpeg 未安装或未添加到 PATH

**解决:**
1. 按第 4 步安装 FFmpeg
2. 确保添加到系统 PATH
3. **重新打开命令提示符**（必须）
4. 验证: `ffmpeg -version`
5. 重启应用

### 问题 5: 端口 5000 已被占用

**错误信息:** `Address already in use`

**解决方案 1: 关闭占用端口的程序**
```bash
# Windows 查找占用 5000 端口的进程
netstat -ano | findstr :5000

# 记下 PID（最后一列数字），然后结束进程
taskkill /PID [进程ID] /F
```

**解决方案 2: 修改端口**

编辑 `app.py` 最后一行：
```python
app.run(host='0.0.0.0', port=8080, debug=True)  # 改为 8080 或其他端口
```

然后访问: http://localhost:8080

### 问题 6: 浏览器无法访问 localhost:5000

**解决:**
1. 确认服务已启动（终端显示 "Running on..."）
2. 尝试访问 http://127.0.0.1:5000
3. 检查防火墙设置
4. 尝试其他浏览器

---

## 🛑 如何停止服务

在运行 Flask 的命令提示符窗口中：

**按 `Ctrl + C`**

你会看到：
```
Keyboard interrupt received, exiting.
```

服务已停止。

---

## 🔄 下次启动（简化步骤）

如果已完成首次配置，下次启动只需：

```bash
# 1. 进入项目目录
cd C:\project2025\qwen235b

# 2. 激活虚拟环境
.\venv\Scripts\activate

# 3. 设置 API Key（如果使用环境变量方式）
set DASHSCOPE_API_KEY=sk-你的API密钥

# 4. 启动
.\venv\Scripts\python app.py
```

或创建一个 `start.bat` 批处理文件：

```batch
@echo off
cd /d C:\project2025\qwen235b
call venv\Scripts\activate
set DASHSCOPE_API_KEY=sk-你的API密钥
python app.py
pause
```

双击 `start.bat` 即可启动！

---

## ✅ 完成！开始使用

1. **选择模型** - 推荐使用 Qwen-VL-Plus
2. **上传视频** - 拖拽或点击上传（最大500MB）
3. **输入问题** - 例如："请描述视频内容"
4. **点击分析** - 等待几秒获得结果

---

## 📋 项目特点

### ✅ 使用 DashScope API 的优势

相比之前的本地模型方案：

| 特性 | DashScope API（新） | 本地模型（旧） |
|-----|-------------------|-------------|
| **模型下载** | ❌ 不需要 | ✅ 需要 200GB+ |
| **GPU 要求** | ❌ 不需要 | ✅ 需要 40GB+ 显存 |
| **安装时间** | 🚀 < 1 分钟 | 🐌 几小时 |
| **硬件成本** | 💰 按使用付费 | 💸 昂贵GPU硬件 |
| **部署难度** | 😊 非常简单 | 😫 复杂配置 |
| **适用设备** | 💻 任何电脑 | 🖥️ 仅高端工作站 |

---

## 🎯 使用示例

### 路测视频分析示例

上传一个行车记录仪视频后，可以这样提问：

```
✅ "请详细描述视频中的道路情况和交通状况。"
✅ "视频中出现了哪些交通标志和信号灯？"
✅ "识别视频中的所有车辆类型。"
✅ "分析视频中是否存在违规驾驶行为。"
✅ "描述视频中的天气和路面状况。"
✅ "总结视频中的关键事件和时间点。"
```

---

## 🔧 常见问题快速解决

### Q: 启动后显示"未配置 API Key"？
**A:** 检查环境变量是否正确设置，或直接在 `app.py` 中配置。

### Q: 提示"未安装 FFmpeg"？
**A:**
1. 按照第 2 步安装 FFmpeg
2. 确保 FFmpeg 已添加到系统 PATH
3. 验证安装: 运行 `ffmpeg -version`
4. 安装后重启应用
5. 查看详细教程: `FFMPEG_INSTALL.md`

### Q: 视频太大无法分析？
**A:**
1. 确保已安装 FFmpeg 并启用自动压缩
2. API 限制: base64 编码后不能超过 10MB
3. 原始视频建议 < 7MB，或由 FFmpeg 自动压缩
4. 可以手动压缩视频或截取较短片段

### Q: 上传视频后报错？
**A:** 检查：
1. 视频格式是否支持（MP4/AVI/MKV/MOV/FLV/WMV）
2. 文件大小是否超过 500MB
3. API Key 是否正确
4. 网络连接是否正常
5. 如果是大文件，确保已安装 FFmpeg

### Q: 如何查看 API 使用量和费用？
**A:** 登录阿里云控制台：https://dashscope.console.aliyun.com/

### Q: 支持哪些视频类型？
**A:** 支持常见视频格式：
- MP4（推荐）
- AVI
- MKV
- MOV
- FLV
- WMV

### Q: FFmpeg 自动压缩需要多久？
**A:**
- 取决于视频大小和电脑性能
- 一般 10-30 秒处理 50MB 视频
- 压缩目标: 约 6.5MB
- 可以在网页上关闭自动压缩功能

---

## 📊 模型对比

### Qwen-VL-Plus（推荐）
- ✅ 性价比最高
- ✅ 速度快
- ✅ 适合日常使用
- 💰 价格适中

### Qwen3-VL-Plus
- ✅ 最新特性
- ✅ 更好的理解能力
- 💰 价格略高

### Qwen-VL-Max
- ✅ 最强性能
- ✅ 复杂场景分析
- ✅ 高精度要求
- 💰 价格最高

---

## 🔐 安全提示

1. **不要分享 API Key** - 避免泄露到公共代码仓库
2. **使用环境变量** - 推荐使用环境变量而不是硬编码
3. **添加 .env 到 .gitignore** - 防止提交敏感信息
4. **定期检查用量** - 避免意外产生高额费用
5. **设置用量限制** - 在阿里云控制台设置预算提醒

---

## 📁 项目文件说明

```
qwen235b/
├── app.py              # ⭐ 主程序（Flask后端）
├── templates/
│   └── index.html     # 🎨 前端界面
├── requirements.txt    # 📦 依赖列表（仅2个包）
├── .env.example       # 🔑 配置模板
├── README.md          # 📖 详细文档
└── QUICKSTART.md      # 🚀 本文件
```

---

## 💡 高级配置

### 自定义端口

编辑 `app.py` 最后一行：
```python
app.run(host='0.0.0.0', port=8080, debug=True)  # 改为 8080
```

### 修改上传大小限制

编辑 `app.py` 第16行：
```python
MAX_VIDEO_SIZE = 1024 * 1024 * 1024  # 改为 1GB
```

### 使用 .env 文件

```bash
# 安装 python-dotenv
pip install python-dotenv

# 创建 .env 文件
copy .env.example .env

# 在 app.py 开头添加
from dotenv import load_dotenv
load_dotenv()
```

---

## 🎓 学习资源

### DashScope API 文档
- 官方文档：https://help.aliyun.com/zh/dashscope/
- 视频理解：https://help.aliyun.com/zh/model-studio/qwenvl-video-understanding
- API参考：https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-api

### Qwen 模型
- GitHub：https://github.com/QwenLM/Qwen3-VL
- ModelScope：https://modelscope.cn/models/Qwen/

---

## 🆘 获取帮助

### 遇到问题？

1. **查看 README.md** - 详细文档
2. **检查控制台输出** - 查看错误信息
3. **访问健康检查** - http://localhost:5000/health
4. **查看配置信息** - http://localhost:5000/config

### 联系方式

- 阿里云 DashScope 支持：https://help.aliyun.com/
- 提交 Issue：在项目 GitHub 页面

---

## 🎉 完成！

现在您已经成功配置了 Qwen3-VL 视频分析系统！

**下一步：**
1. 🎬 上传一个测试视频
2. 💬 尝试不同的问题
3. 🔄 切换不同的模型对比效果
4. 📊 查看分析结果

祝您使用愉快！ 🚗📹🤖
