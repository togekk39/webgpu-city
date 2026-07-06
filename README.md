# webgpu-city

`webgpu-city` 是一個以 Rust、wgpu、winit 與 WebAssembly 撰寫的 3D 城市展示程式。程式保留夕陽天空、陰影與後製效果，城市場景則改由外部 glTF / GLB 模型提供。

## 功能特色

- 使用 `wgpu` 建立 WebGPU render pipeline。
- 使用 `winit` 在 Web 平台建立並掛載 `<canvas>`。
- 從 `WEBGPU_CITY_GLTF_URL` 指定的 glTF / GLB 下載連結或本機路徑載入城市 mesh（原生執行）。
- 支援 GLB 內嵌 PNG base-color 材質貼圖，並沿用現有夕陽場景的材質與光照流程。
- 使用 WGSL shader 呈現夕陽天空、陰影、glow、街道路面與距離霧化效果。

## 專案結構

```text
.
├── Cargo.toml          # Rust crate 與依賴設定
├── Dockerfile.dev      # 開發容器映像設定
├── docker-compose.yml  # Docker Compose 開發環境
├── index.html          # Web 入口頁面，供 Trunk 載入 wasm
├── README.md
└── src
    ├── main.rs         # 應用程式、wgpu 初始化與 glTF / GLB 城市場景載入
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

## 城市模型下載設定（`WEBGPU_CITY_GLTF_URL`）

城市模型不再內建於 repo。原生執行時，啟動前必須用 `WEBGPU_CITY_GLTF_URL` 指定模型來源；此值可以是 HTTP(S) 下載連結，也可以是本機檔案路徑：

```bash
# 從下載連結載入 GLB / glTF
WEBGPU_CITY_GLTF_URL=https://example.com/models/city.glb cargo run

# 或從本機檔案載入
WEBGPU_CITY_GLTF_URL=./models/city.glb cargo run
```

也可以在 repo 根目錄建立 `.env`，讓原生執行與 Trunk / wasm 建置時自動帶入同一個模型來源：

```dotenv
WEBGPU_CITY_GLTF_URL=https://example.com/models/city.glb
```

```bash
cargo run
trunk serve --open
```

WebAssembly / Trunk 執行時沒有 runtime 環境變數；請在 `.env` 或建置環境提供 `WEBGPU_CITY_GLTF_URL`，或在頁面 URL 上加上 `city` / `city_gltf_url` query 參數：

```bash
WEBGPU_CITY_GLTF_URL=https://example.com/models/city.glb trunk serve --open

# 或啟動後開啟類似：
# http://127.0.0.1:3000/?city=https%3A%2F%2Fexample.com%2Fmodels%2Fcity.glb
```

若使用 `WEBGPU_CITY_GLTF_URL=assets/city.glb` 搭配 Trunk，`index.html` 會透過 `data-trunk` copy-dir 將本機 `assets/` 目錄複製到輸出目錄；請確認本機存在 `assets/city.glb`，否則瀏覽器可能抓到 HTML fallback 而不是 GLB。

目前載入器預期模型的 geometry buffer 內嵌在 GLB binary buffer 中；PNG 材質支援 GLB 內嵌的 `image/png` base-color texture、glTF/GLB 內的 `data:image/png;base64,...` URI，以及相對於模型檔案位置的外部 `.png` URI（例如 `assets/city.glb` 旁邊的 `assets/textures/albedo.png`）。若模型使用材質名稱，程式會將常見名稱（例如 `asphalt`、`brick`、`curtain_wall`、`emissive_window`、`metal`、`roof_tar`、`solar`）映射到既有 shader 材質效果；未知材質名稱會以 concrete 材質處理。

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

Docker Compose 會建置包含 `wasm32-unknown-unknown` target 與 Trunk 的開發映像，並在容器中啟動 Web 開發伺服器：

```bash
docker compose up --build dev
```

預設會將容器內的 Trunk `8080` port 對應到主機的 `8080` port，開啟 <http://localhost:8080/> 即可瀏覽。若要改用其他主機 port：

```bash
TRUNK_PORT=3000 docker compose up --build dev
```

若只想在相同容器環境確認編譯：

```bash
docker compose run --rm dev cargo check
docker compose run --rm dev cargo check --target wasm32-unknown-unknown
```

## 授權

請參閱 [LICENSE](LICENSE)。
