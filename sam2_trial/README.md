# AutoSEM · 从描述到编辑

AutoSEM 是一个本地优先的 AI 选区与局部编辑网站：用户可以上传一张图片，用一句自然语言描述希望保留的主体和效果；Qwen 会将需求解析为受限的本地编辑计划、自动定位主体，SAM2 最终生成像素级选区并完成合成。手动分割 Agent、点／框提示和选区画笔仍完整保留。选区完成后，可补画、擦除、收缩／扩展、羽化，并在原图尺寸下导出透明抠图、纯色背景、背景虚化或局部调色版本。

它不是让大模型直接“画边界”：

```text
一句剪辑需求 ──> Qwen 受限编辑计划 ──> Qwen 视觉定位 ──> SAM2 ──> 本地全分辨率合成

手动模式：文字描述 ──可选──> Qwen 视觉定位 ──> Agent 决策 ──> 候选框 / 追问 / 手动提示
                                                          │
点 / 框 ─────────────────────────────────────────────────┼──> SAM2 Tiny ──> 质量复核 ──> 可编辑 Mask
                                                                                          │
选区笔刷 / 背景 / 局部调整 ───────────────────────────────────────────────────────────────┴──> 原图尺寸 PNG
```

当前版本默认使用 **SAM2.1 Tiny + CPU**。分割放进单个后台任务队列：页面会立即返回、显示排队和运行状态，CPU 推理完成后再展示结果。之后迁移到 GPU 时网页和 API 不需要重写，但需要 GPU 主机、驱动和 CUDA 镜像，而不只是修改环境变量。

## 已有页面

- `/`：产品首页，解释工作流和数据边界。
- `/workspace`：一句话一键处理、自动选区、自动定位、手动点/框、后台任务、质量复核、选区微调与局部编辑。
- `/guide`：使用指南。
- `/privacy`：本地处理与可选云端定位的边界说明。
- `/ops`：令牌保护的运营面板，显示匿名会话、请求量、成功率、Qwen 与 SAM2 耗时。
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

手动点选和框选不需要任何外部模型。点击“一键处理”、“自动选区”或“查看推荐位置”时，当前图片和文字才会发送给 `.env` 配置的阿里云百炼 Qwen 视觉模型。一键处理先把文字限制为一个白名单编辑配方，再由独立的 Qwen 定位阶段给 SAM2 提供空间提示；Qwen 不能输出任意代码、路径、URL 或 mask 像素，也不能直接生成轮廓。当前一键能力只覆盖单主体抠图、透明/纯色/模糊背景、边缘优化与局部调色；删除、替换、补图、扩图和全图风格化会明确返回“不支持”，不会假装已经执行。启用内部锚点后，Qwen 在能可靠指出目标内部位置时会额外提供一个包含点，帮助 SAM2 区分框内相邻物体；无法可靠给点时会自动退回为仅使用候选框。

一键处理在每次规划前都会从 `knowledge/one_click_editing.json` 检索与用户需求相关的能力卡，再把它们和固定 JSON 契约提供给 Qwen。这个版本化的本地知识库是后续维护入口：修改已有操作的别名、说明或限制并发布后，Qwen 的理解会随之更新；但它不能凭空开启新功能。要加入真正的新编辑能力，仍必须同时实现服务端合成逻辑、白名单校验与测试，再为它添加知识库条目。

```ini
DASHSCOPE_API_KEY=你的密钥
DASHSCOPE_BASE_URL=你的百炼 OpenAI 兼容地址
DASHSCOPE_MODEL=qwen3-vl-flash
QWEN_REPRESENTATIVE_POINT_ENABLED=true
```

密钥只在后端读取，网页 JavaScript 不会拿到它。不要把 `.env`、密钥或用户图片提交到 Git。

## 输出与保留时间

每次成功任务都会生成：

- `mask.png`：黑白二值掩码；
- `preview.jpg`：轻量的原图、mask 和轮廓叠加预览；
- `contours.png`：透明背景的轮廓；
- `result.json`：原图坐标系中的轮廓、提示、Qwen 候选框（如果使用）与 SAM2 元数据。

在“选区编辑”中生成预览后，还会生成一组派生文件：

- `edited.png`：保持原图尺寸的当前编辑结果；透明背景时为 RGBA PNG；
- `mask.png`：补画、擦除和边缘调整后的选区；
- `preview.png`：用于网页显示的轻量预览；
- `edit.json`：不包含图片像素的编辑配方。

编辑导出从服务器保存的原图与原始 SAM2 mask 重新合成，不会从浏览器的 1600px 显示画布导出。选区笔刷只提交原图坐标、半径和 add/erase 指令；服务器不会接收用户上传的伪造 mask 文件。

上传后，浏览器会立即显示本地预览；服务器则保存原尺寸处理图和一个较小的显示预览。结果面板只加载 `preview.jpg`，而完整 mask、透明轮廓和 JSON 仍通过下载链接提供。预览尺寸由 `DISPLAY_MAX_IMAGE_EDGE` 与 `RESULT_PREVIEW_MAX_EDGE` 控制。

上传图片、任务元数据与结果默认保留 72 小时，可在 `.env` 中通过 `DATA_TTL_HOURS` 修改。设为 `0` 会关闭自动清理；上传触发的清理扫描受 `CLEANUP_INTERVAL_SECONDS` 节流。每个浏览器有自己的签名会话 cookie，不能通过猜测 UUID 访问另一个浏览器的上传或结果；这仍是首版的轻量隔离，公开多用户产品应增加真正的登录和对象存储权限策略。

## 运营面板

`/ops` 是单独的管理员页面，不会出现在普通用户工作区。它以 `X-Ops-Token` 请求头读取聚合数据，浏览器只在当前会话中保存你输入的访问码，访问码不会写入 URL、Git 或用户任务数据。

在服务器 `.env` 设置一个高强度访问码：

```ini
OPS_DASHBOARD_TOKEN=请填随机长字符串
METRICS_RETENTION_DAYS=30
```

然后访问 `/ops` 并输入访问码。面板记录的是匿名会话哈希、事件类型、耗时和结果状态；不会记录图片、文件名、文字描述、Qwen 响应、原始 IP 或访问码。`图片处理`时间只覆盖图片到达服务器后的解码、保存和预览生成，不含网络传输；`SAM2` 面板时间是排队结束后的整段任务时间（读图、预处理、SAM2 和写出结果）；`Qwen` 时间是一次自动定位请求的端到端耗时。统计从本版本部署后开始累积。

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
   若要启用 `/ops`，同时设置 `OPS_DASHBOARD_TOKEN`；不要把该访问码发给普通访客。
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

测试覆盖轮廓导出、Qwen 响应验证、候选框约束、一键编辑配方校验／跨会话隔离／全分辨率 PNG 导出、Agent 的候选选择／手动回退／服务端候选框约束／质量复核，以及后台 job 的提交、轮询、持久化和结果文件访问。
