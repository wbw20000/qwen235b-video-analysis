# FFmpeg 安装指南

## 为什么需要 FFmpeg？

FFmpeg 用于自动压缩大于 7MB 的视频文件，使其符合阿里云 DashScope API 的大小限制（base64 编码后不超过 10MB）。

---

## Windows 安装步骤

### 方法 1: 使用 Chocolatey（推荐）

```bash
# 如果已安装 Chocolatey
choco install ffmpeg

# 验证安装
ffmpeg -version
```

### 方法 2: 手动安装

1. **下载 FFmpeg**
   - 访问：https://www.gyan.dev/ffmpeg/builds/
   - 下载：`ffmpeg-release-essentials.zip`

2. **解压文件**
   - 解压到：`C:\ffmpeg`
   - 确保路径为：`C:\ffmpeg\bin\ffmpeg.exe`

3. **添加到环境变量**

   **方法 A：使用命令行（需要管理员权限）**
   ```bash
   setx /M PATH "%PATH%;C:\ffmpeg\bin"
   ```

   **方法 B：使用图形界面**
   - 右键"此电脑" → "属性"
   - 点击"高级系统设置"
   - 点击"环境变量"
   - 在"系统变量"中找到"Path"
   - 点击"编辑" → "新建"
   - 添加：`C:\ffmpeg\bin`
   - 点击"确定"保存

4. **验证安装**
   ```bash
   # 重新打开命令行窗口
   ffmpeg -version
   ```

---

## Linux 安装

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg

# 验证
ffmpeg -version
```

### CentOS/RHEL
```bash
sudo yum install epel-release
sudo yum install ffmpeg

# 验证
ffmpeg -version
```

---

## macOS 安装

### 使用 Homebrew
```bash
brew install ffmpeg

# 验证
ffmpeg -version
```

---

## 验证 FFmpeg 功能

安装完成后，访问应用的健康检查接口：

```
http://localhost:5000/health
```

应该看到：
```json
{
  "status": "ok",
  "api_configured": true,
  "ffmpeg_installed": true,
  "available_models": [...]
}
```

---

## 使用视频压缩功能

1. **启用自动压缩**
   - 在网页上勾选"自动压缩大视频"选项

2. **上传视频**
   - 上传大于 7MB 的视频文件

3. **自动处理**
   - 系统会自动检测视频大小
   - 如果 > 7MB，自动调用 FFmpeg 压缩到 6.5MB
   - 压缩过程显示在后台日志中

4. **下载压缩视频**
   - 分析完成后，会显示下载按钮
   - 点击即可下载压缩后的视频

---

## 压缩参数说明

当前压缩设置：

- **目标大小**: 6.5 MB
- **音频码率**: 128 kbps
- **视频码率**: 根据视频时长自动计算
- **最低码率**: 100 kbps

压缩算法会根据视频时长自动调整视频码率，确保最终文件大小在目标范围内。

---

## 常见问题

### Q: FFmpeg 安装后还是提示"未安装"？

**A:**
1. 确认已重新打开命令行/终端
2. 运行 `ffmpeg -version` 验证
3. 重启 Flask 应用（重新运行 `start.bat`）

### Q: 压缩需要多长时间？

**A:** 取决于视频时长和文件大小：
- 1 分钟视频：约 10-30 秒
- 5 分钟视频：约 1-2 分钟

### Q: 压缩会降低视频质量吗？

**A:** 会有一定程度的质量损失，但：
- 分辨率保持不变
- 仍然可以进行正常的视频分析
- 如需高质量，建议手动使用专业工具压缩

### Q: 可以调整压缩参数吗？

**A:** 可以！编辑 `app.py`：
```python
# 第 310 行附近
compress_video(filepath, compressed_path, target_size_mb=6.5)
# 改为其他值，如 5.0
```

---

## 无 FFmpeg 的替代方案

如果无法安装 FFmpeg：

1. **手动压缩**
   - 使用在线工具：https://www.videosmaller.com/
   - 目标：压缩到 < 7MB

2. **截取片段**
   - 只上传视频的关键部分
   - 减少视频时长

3. **降低分辨率**
   - 使用视频编辑软件
   - 降低到 720p 或 480p

---

## 技术支持

如果遇到问题：
1. 查看后台日志输出
2. 访问 `/health` 接口检查状态
3. 确认 FFmpeg 版本 >= 4.0

---

**祝您使用愉快！** 🎬
