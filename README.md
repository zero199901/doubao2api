# DOUBAO-WEB-01 / doubao2api

[English](#english) | [中文](#中文) | [Русский](#русский)

## English

DOUBAO-WEB-01 is the maintained Doubao Web reverse-proxy provider used by gen2api.

Repository: https://github.com/carzygod/doubao2api

This project does not use official Volcengine/ARK API keys. It drives logged-in Doubao Web sessions through Playwright and exposes OpenAI-compatible APIs for chat, image generation, music generation, and async video generation.

### Capabilities

| Area | Status |
|---|---|
| Storage | SQLite |
| Redis | Not used |
| Admin WebUI | `/admin?key=<DOUBAO_API_KEY>` |
| Account pool | Multiple Doubao Web accounts |
| Login | Admin WebUI QR/login flow |
| Hot accounts | Keeps a configurable number of Playwright accounts warm |
| Account test | Sends a real upstream chat request and requires model output |
| Account delete | Stops the account browser and removes SQLite rows, usage records, session file, and browser profile for non-default accounts |
| Chat API | `/v1/chat/completions` |
| Image API | `/v1/images/generations` |
| Music API | `/v1/audio/generations` |
| Video API | `/v1/videos`, `/v1/video/generations`, `/v1/videos/generations` |
| Video polling | `/v1/videos/{task_id}` plus legacy aliases |
| Video cancel | `/v1/videos/{task_id}/cancel` plus legacy aliases |
| Quota accounting | Local 24h image/video reservation and completion accounting |
| NewAPI use | Can be added as an OpenAI-compatible channel for chat/image/video |

### Models

| Capability | Model |
|---|---|
| Chat | `doubao` |
| Chat | `doubao-pro` |
| Chat | `doubao-think` |
| Chat | `doubao-expert` |
| Image | `doubao-image` |
| Music | `doubao-music` |
| Video | `doubao-video` |
| Video alias | `seedance_v2.0` |
| Video alias | `seedance2.0` |
| Video alias | `seedance2.0fast` |
| Video alias | `seedance-2.0-fast` |
| Video alias | `seedance_2.0_fast` |
| Video alias | `seedance_v2.0_fast` |
| Video alias | `Seedance 2.0` |
| Video alias | `Seedance 2.0 Fast` |

The recommended NewAPI video model is `doubao-video`. The aliases map to the observed Doubao Web Seedance 2.0 route.

### Quick Start

```bash
docker build -t doubao-web-01:latest .
docker run -d --name doubao-web-01 \
  -p 19090:9090 \
  -e DOUBAO_HOST=0.0.0.0 \
  -e DOUBAO_PORT=9090 \
  -e DOUBAO_API_KEY=change-me-api-key \
  -e DOUBAO_ACCOUNT_DATA_DIR=/app/data \
  -e DOUBAO_BROWSER_DATA=/app/data/browser \
  -v "$PWD/data:/app/data" \
  doubao-web-01:latest
```

## 账户检修（私有 noVNC）

使用 `Dockerfile.novnc` 时必须设置 `VNC_PASSWORD`，并将容器 `6080` 仅映射到宿主机
`127.0.0.1`。`DOUBAO_NOVNC_URL` 是后台展示的私有入口。检修会先写入 SQLite 租约、
阻止新任务，并等待已经借用浏览器的任务结束，再停止该账户的常驻无头浏览器，以
`headless=False` 打开原账户 `user_data_dir`。等待时间由
`DOUBAO_MAINTENANCE_DRAIN_TIMEOUT_SECONDS` 控制，默认 120 秒；超时会释放维护租约并保留
普通浏览器，不会中断仍在执行的任务。租约活动期间账号选择和浏览器启动都会拒绝使用该账户。

接口为 `/admin/api/accounts/{id}/maintenance/{start|heartbeat|stop|validate}`。豆包登录
状态直接持久化到 profile，不需要 Cookie 捕获步骤。默认不清理 Chromium
`Singleton*` 文件；只有显式设置 `DOUBAO_CLEAR_STALE_PROFILE_LOCKS=true` 才允许旧兼容
行为。

Open `http://127.0.0.1:19090/admin?key=change-me-api-key`, add Doubao accounts, run account tests, then route traffic.

### Public API Example

```bash
curl http://127.0.0.1:19090/v1/videos \
  -H "Authorization: Bearer change-me-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao-video",
    "prompt": "a white cube slowly rotating on a desk, realistic photo style",
    "duration": 5,
    "resolution": "720P",
    "ratio": "16:9"
  }'
```

## 中文

DOUBAO-WEB-01 是 gen2api 使用的豆包 Web 反代维护版。

仓库： https://github.com/carzygod/doubao2api

本项目不使用火山引擎 / ARK 官方 API Key，而是通过 Playwright 调度已登录的豆包 Web 会话，对外提供对话、生图、音乐、异步生视频接口，并尽量保持 OpenAI 兼容。

### 能力

| 模块 | 状态 |
|---|---|
| 存储 | SQLite |
| Redis | 不使用 |
| Admin WebUI | `/admin?key=<DOUBAO_API_KEY>` |
| 账号池 | 多个豆包 Web 登录账号 |
| 登录 | Admin WebUI 扫码 / 登录流程 |
| 热账号 | 可配置 Playwright 热账号数量 |
| 账号测试 | 发送真实上游对话请求，拿到模型返回才算成功 |
| 账号删除 | 非默认账号删除时会停止浏览器，并清理 SQLite 记录、使用记录、session 文件和浏览器 profile |
| 对话接口 | `/v1/chat/completions` |
| 生图接口 | `/v1/images/generations` |
| 音乐接口 | `/v1/audio/generations` |
| 生视频接口 | `/v1/videos`、`/v1/video/generations`、`/v1/videos/generations` |
| 视频轮询 | `/v1/videos/{task_id}` 及旧别名 |
| 视频取消 | `/v1/videos/{task_id}/cancel` 及旧别名 |
| 配额统计 | 本地 24 小时图片 / 视频预约与完成计数 |
| NewAPI | 可作为 OpenAI 兼容渠道接入对话 / 图片 / 视频 |

### 模型

| 能力 | 模型 |
|---|---|
| 对话 | `doubao` |
| 对话 | `doubao-pro` |
| 对话 | `doubao-think` |
| 对话 | `doubao-expert` |
| 生图 | `doubao-image` |
| 音乐 | `doubao-music` |
| 生视频 | `doubao-video` |
| 生视频别名 | `seedance_v2.0` |
| 生视频别名 | `seedance2.0` |
| 生视频别名 | `seedance2.0fast` |
| 生视频别名 | `seedance-2.0-fast` |
| 生视频别名 | `seedance_2.0_fast` |
| 生视频别名 | `seedance_v2.0_fast` |
| 生视频别名 | `Seedance 2.0` |
| 生视频别名 | `Seedance 2.0 Fast` |

推荐在 NewAPI 中使用 `doubao-video` 作为视频模型名。其它别名映射到当前观测到的豆包 Web Seedance 2.0 路线。

### 快速启动

```bash
docker build -t doubao-web-01:latest .
docker run -d --name doubao-web-01 \
  -p 19090:9090 \
  -e DOUBAO_HOST=0.0.0.0 \
  -e DOUBAO_PORT=9090 \
  -e DOUBAO_API_KEY=change-me-api-key \
  -e DOUBAO_ACCOUNT_DATA_DIR=/app/data \
  -e DOUBAO_BROWSER_DATA=/app/data/browser \
  -v "$PWD/data:/app/data" \
  doubao-web-01:latest
```

打开 `http://127.0.0.1:19090/admin?key=change-me-api-key`，新增豆包账号并测试通过后再承载业务请求。

### 调用示例

```bash
curl http://127.0.0.1:19090/v1/videos \
  -H "Authorization: Bearer change-me-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao-video",
    "prompt": "一个白色立方体在桌面上缓慢旋转，真实摄影风格",
    "duration": 5,
    "resolution": "720P",
    "ratio": "16:9"
  }'
```

## Русский

DOUBAO-WEB-01 — поддерживаемый Web reverse-proxy для Doubao, используемый в gen2api.

Репозиторий: https://github.com/carzygod/doubao2api

Проект не использует официальные Volcengine/ARK API keys. Он управляет авторизованными Doubao Web-сессиями через Playwright и предоставляет OpenAI-compatible API для chat, image, music и async video generation.

### Возможности

| Area | Status |
|---|---|
| Storage | SQLite |
| Redis | Не используется |
| Admin WebUI | `/admin?key=<DOUBAO_API_KEY>` |
| Account pool | Несколько Doubao Web аккаунтов |
| Login | QR/login flow в Admin WebUI |
| Hot accounts | Настраиваемое число теплых Playwright аккаунтов |
| Account test | Реальный upstream chat request с проверкой ответа модели |
| Account delete | Для non-default аккаунтов останавливает браузер и удаляет SQLite rows, usage records, session file и browser profile |
| Chat API | `/v1/chat/completions` |
| Image API | `/v1/images/generations` |
| Music API | `/v1/audio/generations` |
| Video API | `/v1/videos`, `/v1/video/generations`, `/v1/videos/generations` |
| Video polling | `/v1/videos/{task_id}` и legacy aliases |
| Video cancel | `/v1/videos/{task_id}/cancel` и legacy aliases |
| Quota accounting | Локальный 24h учет image/video |
| NewAPI use | Можно добавить как OpenAI-compatible канал |

### Модели

| Capability | Model |
|---|---|
| Chat | `doubao` |
| Chat | `doubao-pro` |
| Chat | `doubao-think` |
| Chat | `doubao-expert` |
| Image | `doubao-image` |
| Music | `doubao-music` |
| Video | `doubao-video` |
| Video alias | `seedance_v2.0` |
| Video alias | `seedance2.0` |
| Video alias | `seedance2.0fast` |
| Video alias | `seedance-2.0-fast` |
| Video alias | `seedance_2.0_fast` |
| Video alias | `seedance_v2.0_fast` |
| Video alias | `Seedance 2.0` |
| Video alias | `Seedance 2.0 Fast` |

Рекомендуемое имя video-модели для NewAPI: `doubao-video`.

### Быстрый старт

```bash
docker build -t doubao-web-01:latest .
docker run -d --name doubao-web-01 \
  -p 19090:9090 \
  -e DOUBAO_HOST=0.0.0.0 \
  -e DOUBAO_PORT=9090 \
  -e DOUBAO_API_KEY=change-me-api-key \
  -e DOUBAO_ACCOUNT_DATA_DIR=/app/data \
  -e DOUBAO_BROWSER_DATA=/app/data/browser \
  -v "$PWD/data:/app/data" \
  doubao-web-01:latest
```

Откройте `http://127.0.0.1:19090/admin?key=change-me-api-key`, добавьте аккаунты Doubao и выполните account test.

### Пример API

```bash
curl http://127.0.0.1:19090/v1/videos \
  -H "Authorization: Bearer change-me-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao-video",
    "prompt": "a white cube slowly rotating on a desk, realistic photo style",
    "duration": 5,
    "resolution": "720P",
    "ratio": "16:9"
  }'
```
### Multi-Image Reference Video Notes

The video API accepts uploaded Doubao image keys, HTTP(S) image URLs, or `data:image/...;base64,...` data URIs. The server uploads URL/data-URI references to Doubao first, then forwards every resolved key to the web video ability.

Supported request fields include `ref_image_key`, `ref_image_keys`, `reference_image_keys`, `image_url`, `image_urls`, `images`, `reference_images`, `first_frame_image`, and `last_frame_image`.

```bash
curl http://127.0.0.1:19090/v1/videos \
  -H "Authorization: Bearer change-me-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "doubao-video",
    "prompt": "use the two reference frames to create a 5 second transition video",
    "duration": 5,
    "ratio": "16:9",
    "reference_images": [
      {"url": "https://example.com/frame-1.png"},
      {"url": "https://example.com/frame-2.png"}
    ]
  }'
```
