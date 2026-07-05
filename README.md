# webgpu-city

`webgpu-city` 是一個以 Rust、wgpu 與 winit 撰寫的桌面 3D 城市展示程式。程式會產生簡易城市網格、道路與建築物，並透過 WebGPU 後端持續旋轉相機呈現畫面。

## 功能特色

- 使用 `wgpu` 建立 GPU render pipeline。
- 使用 `winit` 建立原生視窗與事件迴圈。
- 以程式化方式產生城市幾何、道路與高樓。
- 使用 WGSL shader 呈現高度 glow、街道路面與距離霧化效果。

## 專案結構

```text
.
├── Cargo.toml          # Rust crate 與依賴設定
├── Dockerfile.dev      # 開發容器映像設定
├── docker-compose.yml  # Docker Compose 開發環境
├── README.md
└── src
    ├── main.rs         # 應用程式、wgpu 初始化與城市 mesh 產生
    └── shader.wgsl     # WGSL vertex/fragment shader
```

## 需求

### 本機開發

- Rust toolchain（建議使用最新版 stable）
- 支援 Vulkan、Metal、DX12 或 GL 的 GPU/驅動程式
- Linux 桌面環境需具備 X11 或 Wayland 顯示環境

### Docker 開發

- Docker Engine
- Docker Compose v2
- Linux 主機上的 GPU/顯示環境
  - X11：需要允許容器連線至 X server
  - Mesa/Vulkan：建議主機已正確安裝 GPU 驅動

> 注意：Docker 設定主要面向 Linux 桌面開發環境。macOS 與 Windows 的 GUI/GPU passthrough 設定差異較大，可能需要依照 Docker Desktop 與顯示伺服器配置額外調整。

## 本機執行

```bash
cargo run
```

若只想確認專案可編譯：

```bash
cargo check
```

## 使用 Docker Compose 啟動開發環境

第一次啟動前，如果使用 X11，通常需要允許本機 Docker 使用者連線至 X server：

```bash
xhost +local:docker
```

建置並執行開發容器：

```bash
docker compose up --build dev
```

Compose 服務會：

- 使用 `Dockerfile.dev` 建立包含 Rust、Mesa/Vulkan 與 Linux 視窗相關開發套件的映像。
- 將目前專案掛載至 `/workspace/webgpu-city`。
- 使用 named volumes 快取 Cargo registry、Git 依賴與 `target` 編譯輸出。
- 掛載 `/tmp/.X11-unix` 與 `/dev/dri`，讓容器可連接主機顯示與 GPU 裝置。
- 預設執行 `cargo run`。

若只想在容器中執行檢查：

```bash
docker compose run --rm dev cargo check
```

若要開啟互動 shell：

```bash
docker compose run --rm dev bash
```

結束後可視需要撤銷 X11 存取權：

```bash
xhost -local:docker
```

## 常見問題

### 容器無法開啟視窗

請確認：

1. 主機的 `DISPLAY` 環境變數存在。
2. 已執行 `xhost +local:docker`（X11 環境）。
3. `/tmp/.X11-unix` 可被容器掛載。
4. Docker Compose 服務有掛載 `/dev/dri`。

### 找不到 Vulkan adapter 或 GPU adapter

請確認主機 GPU 驅動可用，並可在容器中執行：

```bash
docker compose run --rm dev vulkaninfo --summary
```

若 Vulkan 不可用，可嘗試改用其他 wgpu 後端，例如：

```bash
WGPU_BACKEND=gl docker compose up --build dev
```

## 授權

請參閱 [LICENSE](LICENSE)。
