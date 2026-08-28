# AutoSEM · 从描述到轮廓

AutoSEM 是一个本地优先的图片分割网站：用户上传一张图片并描述目标，Qwen（可选）先给出候选框，SAM2 再根据候选框、包含点、排除点或手动框生成像素级轮廓。

它不是让大模型直接“画边界”：

```text
文字描述 ──可选──> Qwen 视觉定位 ──> 候选框
                                       │
点 / 框 ──────────────────────────────┼──> SAM2 Tiny ──> Mask + 轮廓 PNG + JSON
```

当前版本默认使用 **SAM2.1 Tiny + CPU**。分割放进单个后台任务队列：页面会立即返回、显示排队和运行状态，CPU 推理完成后再展示结果。之后迁移到 GPU 时网页和 API 不需要重写，但需要 GPU 主机、驱动和 CUDA 镜像，而不只是修改环境变量。

## 已有页面

- `/`：产品首页，解释工作流和数据边界。
- `/workspace`：上传、Qwen 自动定位、手动点/框、后台任务与结果下载。
- `/guide`：使用指南。
- `/privacy`：本地处理与可选云端定位的边界说明。
- `/healthz`、`/readyz`：部署健康检查。
- `/api/runtime/status`：不暴露路径或密钥的本地引擎状态。

## 本地启动

项目会复用 AutoSEM 已验证过的 SAM2 源码和 Python，不创建新的虚拟环境。

先确认 Tiny 权重在下面默认位置，或在 `.env` 中把 `SAM2_CHECKPOINT` 指向你的实际文件：

```text
C:\Users\11609\Documents\Autosem\models\sam2.1_hiera_tiny.pt
```

如果还没有 `.env`，复制示例；如果已经配置了百炼密钥，只需把示例里缺少的 `SAM2_...` 配置补到原有 `.env`，不要覆盖已有密钥。

```powershell
Copy-Item '.env.example' '.env'
notepad '.env'
```

最关键的 Tiny 配对是：

```ini
SAM2_DEVICE=cpu
SAM2_MODEL_VARIANT=tiny
SAM2_MODEL_NAME=sam2.1_hiera_tiny
SAM2_CHECKPOINT=C:\Users\11609\Documents\Autosem\models\sam2.1_hiera_tiny.pt
SAM2_MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
```

启动：

```powershell
Push-Location 'C:\Users\11609\Documents\ChatGPT\机器学习\sam2_trial'
& 'C:\Users\11609\AppData\Local\Python\pythoncore-3.14-64\python.exe' '.\app.py'
Pop-Location
```

打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。本地 Flask 只绑定 `127.0.0.1`，不会自行对局域网或公网开放。

### CPU 设置

`SAM2_MAX_IMAGE_EDGE=1280` 是 CPU 的输入保护：超大图片会先缩小，再映射 mask 回原图坐标。因此输出 JSON 和 PNG 坐标仍是原图坐标。SAM2 Tiny 的内部编码尺寸仍为 1024，通常不应把这个值降到 1024 以下来换取速度，因为主要会损失输入细节。

`SAM2_MAX_QUEUE=8` 表示最多积压 8 个任务；单 worker 是刻意设计，避免同一台 CPU 同时加载多个 SAM2 实例。Tiny 权重第一次运行会有加载时间，之后会复用内存中的模型。

若要使用 NVIDIA GPU：

```ini
SAM2_DEVICE=cuda
SAM2_MAX_IMAGE_EDGE=0
```

GPU 可加速，但仍只有一个推理 worker；这避免多用户并发时多份模型抢显存。

## Qwen 自动定位（可选）

手动点选和框选不需要任何外部模型。只有点击“仅自动定位”或“自动定位并生成轮廓”时，当前图片和文字描述才会发送给 `.env` 配置的阿里云百炼 Qwen 视觉模型。

```ini
DASHSCOPE_API_KEY=你的密钥
DASHSCOPE_BASE_URL=你的百炼 OpenAI 兼容地址
DASHSCOPE_MODEL=qwen3-vl-flash
```

密钥只在后端读取，网页 JavaScript 不会拿到它。不要把 `.env`、密钥或用户图片提交到 Git。

## 输出与保留时间

每次成功任务都会生成：

- `mask.png`：黑白二值掩码；
- `preview.jpg`：轻量的原图、mask 和轮廓叠加预览；
- `contours.png`：透明背景的轮廓；
- `result.json`：原图坐标系中的轮廓、提示、Qwen 候选框（如果使用）与 SAM2 元数据。

上传后，浏览器会立即显示本地预览；服务器则保存原尺寸处理图和一个较小的显示预览。结果面板只加载 `preview.jpg`，而完整 mask、透明轮廓和 JSON 仍通过下载链接提供。预览尺寸由 `DISPLAY_MAX_IMAGE_EDGE` 与 `RESULT_PREVIEW_MAX_EDGE` 控制。

上传图片、任务元数据与结果默认保留 72 小时，可在 `.env` 中通过 `DATA_TTL_HOURS` 修改。设为 `0` 会关闭自动清理；上传触发的清理扫描受 `CLEANUP_INTERVAL_SECONDS` 节流。每个浏览器有自己的签名会话 cookie，不能通过猜测 UUID 访问另一个浏览器的上传或结果；这仍是首版的轻量隔离，公开多用户产品应增加真正的登录和对象存储权限策略。

## 容器化与服务器部署

`Dockerfile`、`docker-compose.yml` 和 `nginx/default.conf` 已准备好 CPU 首版部署：

- CPU-only PyTorch 与固定 SAM2 上游版本；
- Tiny 权重运行时只读挂载，不写进镜像层；
- 一个 Gunicorn worker、两个请求线程，对应一个 SAM2 进程与任务队列；
- Nginx 限制上传大小、直接缓存压缩静态资源，并为 CPU 推理保留足够的反向代理超时；
- 数据使用 Docker volume 持久化。

服务器上应执行以下准备工作：

1. 把 Tiny 权重放到 `sam2_trial/models/sam2.1_hiera_tiny.pt`。
2. 在服务器的 `.env` 里设置 Qwen 密钥（如需要）和高强度 `APP_SECRET_KEY`。
3. 确认 `docker compose` 可用后，在 `sam2_trial` 目录运行：

```bash
docker compose up --build -d
```

4. 域名、HTTPS 证书、ICP备案（若使用中国内地公网域名）和真实用户登录应在正式公开前再配置。

这个仓库只提供可部署骨架，没有替你购买域名、开通服务器或把服务暴露到公网。

## 检查

```powershell
Push-Location 'C:\Users\11609\Documents\ChatGPT\机器学习\sam2_trial'
& 'C:\Users\11609\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m unittest discover -v
Pop-Location
```

测试覆盖轮廓导出、Qwen 响应验证、候选框约束，以及后台 job 的提交、轮询、持久化和结果文件访问。
