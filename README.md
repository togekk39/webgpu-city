# webgpu-city

`webgpu-city` 是一個以 Rust、wgpu、winit 與 WebAssembly 撰寫的 Web 版 3D 城市展示程式。程式會產生簡易城市網格、道路與建築物，並在瀏覽器的 WebGPU canvas 中持續旋轉相機呈現畫面。

## 功能特色

- 使用 `wgpu` 建立 WebGPU render pipeline。
- 使用 `winit` 在 Web 平台建立並掛載 `<canvas>`。
- 以程式化方式產生城市幾何、道路與高樓。
- 使用 WGSL shader 呈現高度 glow、街道路面與距離霧化效果。

## 專案結構

```text
.
├── Cargo.toml          # Rust crate 與依賴設定
├── Dockerfile.dev      # 開發容器映像設定
├── docker-compose.yml  # Docker Compose 開發環境
├── index.html          # Web 入口頁面，供 Trunk 載入 wasm
├── README.md
└── src
    ├── main.rs         # Web 應用程式、wgpu 初始化與城市 mesh 產生
    └── shader.wgsl     # WGSL vertex/fragment shader
```

## 需求

- Rust toolchain（建議使用最新版 stable）
- `wasm32-unknown-unknown` target
- Trunk
- 支援 WebGPU 的瀏覽器（例如新版 Chrome、Edge、Firefox Nightly 或 Safari Technology Preview）

安裝 WebAssembly target 與 Trunk：

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk
```

## Web 版執行

啟動本機開發伺服器：

```bash
trunk serve --open
```

若只想建置靜態檔案：

```bash
trunk build --release
```

建置結果會輸出至 `dist/`，可部署到任何靜態網站服務。

若只想確認原生目標可編譯：

```bash
cargo check
```

若想確認 WebAssembly 目標可編譯：

```bash
cargo check --target wasm32-unknown-unknown
```

## WebGPU 注意事項

- 瀏覽器與作業系統必須啟用 WebGPU。
- 若頁面沒有畫面，請先查看瀏覽器 console 是否有 WebGPU adapter 或 device 建立失敗訊息。
- 部分瀏覽器可能需要透過 `https://` 或 `localhost` 才允許完整 WebGPU 功能。

## Docker 開發

Docker 設定保留為開發環境輔助用途；Web 版主要建議直接使用 Trunk 啟動：

```bash
docker compose run --rm dev cargo check
```

## 授權

請參閱 [LICENSE](LICENSE)。
