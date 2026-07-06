use cgmath::{Deg, InnerSpace, Matrix4, Point3, SquareMatrix, Vector3, ortho, perspective};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{cell::RefCell, rc::Rc, sync::Arc};
use wgpu::util::DeviceExt;
#[cfg(target_arch = "wasm32")]
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};

#[cfg(target_arch = "wasm32")]
fn browser_viewport_size(window: &Window) -> Option<PhysicalSize<u32>> {
    let global = js_sys::global();
    let width = js_sys::Reflect::get(&global, &"innerWidth".into())
        .ok()?
        .as_f64()?;
    let height = js_sys::Reflect::get(&global, &"innerHeight".into())
        .ok()?
        .as_f64()?;
    let scale_factor = window.scale_factor();

    Some(PhysicalSize::new(
        (width * scale_factor).round().max(1.0) as u32,
        (height * scale_factor).round().max(1.0) as u32,
    ))
}

#[cfg(target_arch = "wasm32")]
fn sync_canvas_to_viewport(window: &Window) -> PhysicalSize<u32> {
    let size = browser_viewport_size(window).unwrap_or_else(|| window.inner_size());
    let _ = window.request_inner_size(size);
    size
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    material_id: f32,
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 44,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera_position: [f32; 4],
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    light_view_proj: [[f32; 4]; 4],
    sky_color: [f32; 4],
    horizon_color: [f32; 4],
    ambient_color: [f32; 4],
    settings: [f32; 4],
    sunset: [f32; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            inv_view_proj: Matrix4::identity().into(),
            camera_position: [0.0, 0.0, 0.0, 1.0],
            sun_direction: sunset_sun_direction().extend(0.0).into(),
            sun_color: [1.0, 0.61, 0.25, 1.0],
            light_view_proj: Matrix4::identity().into(),
            sky_color: [0.045, 0.085, 0.18, 1.0],
            horizon_color: [1.0, 0.47, 0.15, 1.0],
            ambient_color: [0.095, 0.13, 0.22, 1.0],
            settings: [0.0, 1.12, 0.034, 1.0],
            sunset: [2.0_f32.to_radians(), 220.0_f32.to_radians(), 0.0105, 1.25],
        }
    }
}

struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
const CITY_URL_ENV: &str = "WEBGPU_CITY_GLTF_URL";

#[cfg(not(target_arch = "wasm32"))]
const CITY_URL: Option<&str> = option_env!("WEBGPU_CITY_GLTF_URL");

fn gltf_material(name: &str) -> ([f32; 3], f32) {
    match name {
        "asphalt" => ([0.018, 0.018, 0.017], MAT_ASPHALT),
        "brick" => ([0.25, 0.12, 0.08], MAT_BRICK),
        "curtain_wall" => ([0.035, 0.075, 0.105], MAT_CURTAIN_WALL),
        "emissive_window" => ([0.78, 0.42, 0.18], MAT_EMISSIVE_WINDOW),
        "glass" | "window" => ([0.025, 0.045, 0.062], MAT_WINDOW_PANE),
        "metal" => ([0.18, 0.18, 0.17], MAT_METAL),
        "roof_tar" => ([0.04, 0.038, 0.035], MAT_ROOF_TAR),
        "solar" => ([0.025, 0.035, 0.045], MAT_SOLAR),
        _ => ([0.28, 0.26, 0.23], MAT_CONCRETE),
    }
}

struct MaterialTexture {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl MaterialTexture {
    fn sample(&self, uv: [f32; 2]) -> [f32; 3] {
        let u = uv[0].rem_euclid(1.0);
        let v = uv[1].rem_euclid(1.0);
        let x = (u * self.width as f32)
            .floor()
            .clamp(0.0, self.width.saturating_sub(1) as f32) as u32;
        let y = ((1.0 - v) * self.height as f32)
            .floor()
            .clamp(0.0, self.height.saturating_sub(1) as f32) as u32;
        let offset = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[offset] as f32 / 255.0,
            self.pixels[offset + 1] as f32 / 255.0,
            self.pixels[offset + 2] as f32 / 255.0,
        ]
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download_city_gltf_bytes() -> Vec<u8> {
    let source = std::env::var(CITY_URL_ENV)
        .ok()
        .or_else(|| CITY_URL.map(str::to_owned))
        .expect("set WEBGPU_CITY_GLTF_URL to a .glb/.gltf download URL or local file path");

    if source.starts_with("http://") || source.starts_with("https://") {
        let response = reqwest::blocking::get(&source).expect("download city glTF/GLB");
        response
            .error_for_status()
            .expect("city glTF/GLB download returned an error status")
            .bytes()
            .expect("read city glTF/GLB download body")
            .to_vec()
    } else {
        std::fs::read(&source).expect("read city glTF/GLB file path")
    }
}

#[cfg(target_arch = "wasm32")]
fn download_city_gltf_bytes() -> Vec<u8> {
    panic!("WEBGPU_CITY_GLTF_URL downloads are currently supported by the native build");
}

fn decode_embedded_png_texture(bytes: &[u8]) -> Option<MaterialTexture> {
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .ok()?
        .to_rgba8();
    let (width, height) = image.dimensions();
    Some(MaterialTexture {
        pixels: image.into_raw(),
        width,
        height,
    })
}

fn embedded_image_bytes<'a>(image: gltf::image::Image<'a>, blob: &'a [u8]) -> Option<&'a [u8]> {
    match image.source() {
        gltf::image::Source::View { view, mime_type } if mime_type == "image/png" => {
            let start = view.offset();
            let end = start + view.length();
            blob.get(start..end)
        }
        _ => None,
    }
}

fn load_city_gltf_mesh() -> Mesh {
    let bytes = download_city_gltf_bytes();
    load_city_gltf_mesh_from_slice(&bytes)
}

fn load_city_gltf_mesh_from_slice(bytes: &[u8]) -> Mesh {
    let gltf = gltf::Gltf::from_slice(bytes).expect("parse downloaded city glTF/GLB");
    let blob = gltf
        .blob
        .as_deref()
        .expect("downloaded city GLB must contain an embedded binary buffer");
    let images: Vec<Option<MaterialTexture>> = gltf
        .images()
        .map(|image| embedded_image_bytes(image, blob).and_then(decode_embedded_png_texture))
        .collect();
    let mut mesh = Mesh::new();

    for gltf_mesh in gltf.meshes() {
        for primitive in gltf_mesh.primitives() {
            let reader = primitive.reader(|buffer| (buffer.index() == 0).then_some(blob));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Option<Vec<[f32; 3]>> =
                reader.read_normals().map(|normals| normals.collect());
            let texcoords: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(0)
                .map(|texcoords| texcoords.into_f32().collect());
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|indices| indices.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            let material_doc = primitive.material();
            let material = material_doc
                .name()
                .map(gltf_material)
                .unwrap_or_else(|| gltf_material("concrete"));
            let texture = material_doc
                .pbr_metallic_roughness()
                .base_color_texture()
                .and_then(|info| images.get(info.texture().source().index()))
                .and_then(Option::as_ref);

            for triangle in indices.chunks_exact(3) {
                let p0: Vector3<f32> = positions[triangle[0] as usize].into();
                let p1: Vector3<f32> = positions[triangle[1] as usize].into();
                let p2: Vector3<f32> = positions[triangle[2] as usize].into();
                let face_normal = (p1 - p0).cross(p2 - p0).normalize();
                let base = mesh.vertices.len() as u32;

                for index in triangle {
                    let index = *index as usize;
                    let normal = normals
                        .as_ref()
                        .and_then(|normals| normals.get(index).copied())
                        .unwrap_or_else(|| face_normal.into());
                    let uv = texcoords
                        .as_ref()
                        .and_then(|texcoords| texcoords.get(index).copied())
                        .unwrap_or([positions[index][0], positions[index][2]]);
                    let texture_color = texture
                        .map(|texture| texture.sample(uv))
                        .unwrap_or(material.0);
                    mesh.vertices.push(Vertex {
                        position: positions[index],
                        color: texture_color,
                        normal,
                        uv,
                        material_id: material.1,
                    });
                }

                mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
            }
        }
    }

    mesh
}

struct DepthTexture {
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

struct ShadowTexture {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
}

impl ShadowTexture {
    fn create(device: &wgpu::Device, size: u32) -> Self {
        let format = wgpu::TextureFormat::Depth32Float;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sunset directional shadow map"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sunset shadow comparison sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            sampler,
            format,
        }
    }
}

impl DepthTexture {
    fn create(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let format = wgpu::TextureFormat::Depth24Plus;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("city depth texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            format,
        }
    }
}

struct HdrTarget {
    view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    format: wgpu::TextureFormat,
}

impl HdrTarget {
    fn create(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let format = wgpu::TextureFormat::Rgba16Float;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cinematic hdr scene target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr post sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        Self {
            view,
            _sampler: sampler,
            bind_group,
            format,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderOutcome {
    Presented,
    Lost,
    Outdated,
    Timeout,
    Occluded,
    Validation,
}

#[cfg(not(target_arch = "wasm32"))]
struct AppClock {
    start: Instant,
}

#[cfg(not(target_arch = "wasm32"))]
impl AppClock {
    fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    fn elapsed_secs(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }
}

#[cfg(target_arch = "wasm32")]
struct AppClock {
    start_ms: f64,
}

#[cfg(target_arch = "wasm32")]
impl AppClock {
    fn new() -> Self {
        Self {
            start_ms: js_sys::Date::now(),
        }
    }

    fn elapsed_secs(&self) -> f32 {
        ((js_sys::Date::now() - self.start_ms) / 1000.0) as f32
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sky_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    render_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    post_layout: wgpu::BindGroupLayout,
    hdr: HdrTarget,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    shadow_bind_group: wgpu::BindGroup,
    depth: DepthTexture,
    shadow: ShadowTexture,
    render_scale: f32,
    clock: AppClock,
    camera: CameraController,
    paused_elapsed: Option<f32>,
    elapsed_offset: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let size = sync_canvas_to_viewport(&window);
        #[cfg(not(target_arch = "wasm32"))]
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create window surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("find a GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("create device");
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let render_scale = configured_render_scale();
        let hdr_width = scaled_extent(config.width, render_scale);
        let hdr_height = scaled_extent(config.height, render_scale);

        let mesh = build_city_mesh();
        eprintln!(
            "city stats: vertices={} indices={} draw_calls=4 shadow_map=2048(default)/1024(fallback) render_scale={:0.2} postprocess=targeted_bloom+filmic",
            mesh.vertices.len(),
            mesh.indices.len(),
            render_scale
        );
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("city vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("city indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let mut uniforms = Uniforms::new();
        let camera = CameraController::new();
        let (initial_view_proj, initial_eye) = camera.matrix(config.width, config.height, 0.0);
        uniforms.view_proj = initial_view_proj.into();
        uniforms.inv_view_proj = initial_view_proj
            .invert()
            .unwrap_or_else(Matrix4::identity)
            .into();
        uniforms.camera_position = [initial_eye.x, initial_eye.y, initial_eye.z, 1.0];
        uniforms.light_view_proj = light_matrix().into();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let depth = DepthTexture::create(&device, hdr_width, hdr_height);
        let shadow_size = configured_shadow_size();
        let shadow = ShadowTexture::create(&device, shadow_size);
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow bind group"),
            layout: &shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow.sampler),
                },
            ],
        });
        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let hdr = HdrTarget::create(&device, hdr_width, hdr_height, &post_layout);
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("city pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&shadow_layout)],
            immediate_size: 0,
        });
        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout)],
            immediate_size: 0,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("procedural sunset sky pipeline"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: hdr.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow pipeline layout"),
                bind_group_layouts: &[Some(&uniform_layout)],
                immediate_size: 0,
            });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("directional shadow depth pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                buffers: &[Some(Vertex::layout())],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: shadow.format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("city render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(Vertex::layout())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: hdr.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth.format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_layout),
                Some(&shadow_layout),
                Some(&post_layout),
            ],
            immediate_size: 0,
        });
        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tone map and bloom post pipeline"),
            layout: Some(&post_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_post"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            window,
            surface,
            device,
            queue,
            config,
            sky_pipeline,
            shadow_pipeline,
            render_pipeline,
            post_pipeline,
            post_layout,
            hdr,
            vertex_buffer,
            index_buffer,
            num_indices: mesh.indices.len() as u32,
            uniform_buffer,
            uniform_bind_group,
            shadow_bind_group,
            depth,
            shadow,
            render_scale,
            clock: AppClock::new(),
            camera,
            paused_elapsed: None,
            elapsed_offset: 0.0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let hdr_width = scaled_extent(self.config.width, self.render_scale);
        let hdr_height = scaled_extent(self.config.height, self.render_scale);
        self.depth = DepthTexture::create(&self.device, hdr_width, hdr_height);
        self.hdr = HdrTarget::create(&self.device, hdr_width, hdr_height, &self.post_layout);
    }

    fn update(&self) {
        let elapsed = self.display_elapsed();
        let (view_proj, eye) = self
            .camera
            .matrix(self.config.width, self.config.height, elapsed);
        let mut uniforms = Uniforms::new();
        uniforms.view_proj = view_proj.into();
        uniforms.inv_view_proj = view_proj.invert().unwrap_or_else(Matrix4::identity).into();
        uniforms.camera_position = [eye.x, eye.y, eye.z, 1.0];
        uniforms.light_view_proj = light_matrix().into();
        uniforms.settings[0] = elapsed;
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn display_elapsed(&self) -> f32 {
        self.paused_elapsed
            .unwrap_or_else(|| self.clock.elapsed_secs() - self.elapsed_offset)
    }

    fn toggle_motion_pause(&mut self) {
        if let Some(paused_elapsed) = self.paused_elapsed.take() {
            self.elapsed_offset += self.clock.elapsed_secs() - self.elapsed_offset - paused_elapsed;
        } else {
            self.paused_elapsed = Some(self.display_elapsed());
            self.camera.cancel_pointer_motion();
        }
    }

    fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        if self.paused_elapsed.is_none() {
            self.camera.handle_cursor_moved(position);
        }
    }

    fn handle_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left && state == ElementState::Pressed {
            self.toggle_motion_pause();
        }

        if self.paused_elapsed.is_none() {
            self.camera.handle_mouse_input(button, state);
        }
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        if self.paused_elapsed.is_none() {
            self.camera.handle_mouse_wheel(delta);
        }
    }

    fn render(&mut self) -> RenderOutcome {
        self.update();
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost => return RenderOutcome::Lost,
            wgpu::CurrentSurfaceTexture::Outdated => return RenderOutcome::Outdated,
            wgpu::CurrentSurfaceTexture::Timeout => return RenderOutcome::Timeout,
            wgpu::CurrentSurfaceTexture::Occluded => return RenderOutcome::Occluded,
            wgpu::CurrentSurfaceTexture::Validation => return RenderOutcome::Validation,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("city encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("directional shadow depth pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("procedural sky render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("city render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.shadow_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("postprocess render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.shadow_bind_group, &[]);
            pass.set_bind_group(2, &self.hdr.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        RenderOutcome::Presented
    }
}

fn configured_shadow_size() -> u32 {
    match option_env!("WEBGPU_CITY_SHADOW_SIZE") {
        Some("1024") => 1024,
        _ => 2048,
    }
}

fn configured_render_scale() -> f32 {
    match option_env!("WEBGPU_CITY_RENDER_SCALE") {
        Some("0.7") | Some("0.70") => 0.70,
        Some("0.85") => 0.85,
        _ => 1.0,
    }
}

fn scaled_extent(value: u32, scale: f32) -> u32 {
    ((value as f32 * scale).round() as u32).max(1)
}

#[derive(Clone, Copy)]
enum CameraMode {
    RooftopVista,
    CinematicOrbit,
    HeroStreet,
}

struct CameraController {
    yaw_offset: f32,
    pitch_offset: f32,
    pan: Vector3<f32>,
    zoom: f32,
    last_cursor: Option<PhysicalPosition<f64>>,
    left_dragging: bool,
}

impl CameraController {
    const ROTATION_SENSITIVITY: f32 = 0.003;
    const PAN_SENSITIVITY: f32 = 0.018;
    const MIN_CAMERA_Y: f32 = 0.25;
    const MIN_PITCH: f32 = -0.35;
    const MAX_PITCH: f32 = 0.85;
    const MIN_ZOOM: f32 = 0.35;
    const MAX_ZOOM: f32 = 2.4;
    const WHEEL_ZOOM_SENSITIVITY: f32 = 0.11;

    fn new() -> Self {
        Self {
            yaw_offset: 0.0,
            pitch_offset: 0.0,
            pan: Vector3::new(0.0, 0.0, 0.0),
            zoom: 1.0,
            last_cursor: None,
            left_dragging: false,
        }
    }

    fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        if let Some(last) = self.last_cursor {
            let delta_x = (position.x - last.x) as f32;
            let delta_y = (position.y - last.y) as f32;

            if self.left_dragging {
                let (_, target) = camera_base_pose(0.0);
                let eye = self.eye_for_target(target);
                let forward = (target - eye).normalize();
                let right = forward.cross(Vector3::unit_y()).normalize();
                self.pan += right * (-delta_x * Self::PAN_SENSITIVITY);
                self.pan += Vector3::unit_y() * (-delta_y * Self::PAN_SENSITIVITY);
                self.clamp_pan_to_horizon();
            } else {
                self.yaw_offset += delta_x * Self::ROTATION_SENSITIVITY;
                self.pitch_offset = (self.pitch_offset - delta_y * Self::ROTATION_SENSITIVITY)
                    .clamp(Self::MIN_PITCH, Self::MAX_PITCH);
            }
        }

        self.last_cursor = Some(position);
    }

    fn handle_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.left_dragging = state == ElementState::Pressed;
            if state == ElementState::Released {
                self.last_cursor = None;
            }
        }
    }

    fn cancel_pointer_motion(&mut self) {
        self.left_dragging = false;
        self.last_cursor = None;
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let scroll_lines = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(position) => position.y as f32 / 120.0,
        };
        let zoom_factor = (1.0 - scroll_lines * Self::WHEEL_ZOOM_SENSITIVITY).max(0.05);
        self.zoom = (self.zoom * zoom_factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
    }

    fn matrix(&self, width: u32, height: u32, time: f32) -> (Matrix4<f32>, Point3<f32>) {
        let aspect = width as f32 / height as f32;
        let (base_eye, base_target) = camera_base_pose(time);
        let mut target = base_target + self.pan;
        let mut eye = self.eye_for_target_from(base_eye, base_target) + self.pan;
        if eye.y < Self::MIN_CAMERA_Y {
            let correction = Self::MIN_CAMERA_Y - eye.y;
            eye.y += correction;
            target.y += correction;
        }
        let view = Matrix4::look_at_rh(eye, target, Vector3::unit_y());
        let proj = perspective(Deg(camera_fov()), aspect, 0.1, 220.0);
        (proj * view, eye)
    }

    fn eye_for_target(&self, target: Point3<f32>) -> Point3<f32> {
        let (base_eye, base_target) = camera_base_pose(0.0);
        self.eye_for_target_from(base_eye, base_target) + (target - base_target)
    }

    fn eye_for_target_from(&self, base_eye: Point3<f32>, base_target: Point3<f32>) -> Point3<f32> {
        let offset = base_eye - base_target;
        let base_radius = offset.magnitude();
        let radius = base_radius * self.zoom;
        let base_yaw = offset.x.atan2(offset.z);
        let base_pitch = (offset.y / base_radius).asin();
        let yaw = base_yaw + self.yaw_offset;
        let pitch = (base_pitch + self.pitch_offset).clamp(Self::MIN_PITCH, Self::MAX_PITCH);
        let horizontal = radius * pitch.cos();

        Point3::new(
            base_target.x + horizontal * yaw.sin(),
            base_target.y + radius * pitch.sin(),
            base_target.z + horizontal * yaw.cos(),
        )
    }

    fn clamp_pan_to_horizon(&mut self) {
        let (_, base_target) = camera_base_pose(0.0);
        let eye = self.eye_for_target(base_target) + self.pan;

        if eye.y < Self::MIN_CAMERA_Y {
            self.pan.y += Self::MIN_CAMERA_Y - eye.y;
        }
    }
}

fn sunset_sun_direction() -> Vector3<f32> {
    let elevation = 2.0_f32.to_radians();
    let azimuth = 220.0_f32.to_radians();
    Vector3::new(
        azimuth.cos() * elevation.cos(),
        elevation.sin(),
        azimuth.sin() * elevation.cos(),
    )
    .normalize()
}

fn light_matrix() -> Matrix4<f32> {
    let sun = sunset_sun_direction();
    let center = Point3::new(0.0, 5.0, -24.0);
    let eye = center + sun * 82.0;
    let view = Matrix4::look_at_rh(eye, center, Vector3::unit_y());
    let proj = ortho(-58.0, 58.0, -42.0, 50.0, 1.0, 170.0);
    proj * view
}

fn camera_mode() -> CameraMode {
    match option_env!("WEBGPU_CITY_CAMERA") {
        Some("orbit") => CameraMode::CinematicOrbit,
        Some("hero") | Some("street") => CameraMode::HeroStreet,
        _ => CameraMode::RooftopVista,
    }
}

fn camera_fov() -> f32 {
    match camera_mode() {
        CameraMode::RooftopVista => 42.0,
        _ => 50.0,
    }
}

fn camera_base_pose(time: f32) -> (Point3<f32>, Point3<f32>) {
    match camera_mode() {
        CameraMode::RooftopVista => (Point3::new(10.5, 8.8, 18.5), Point3::new(-5.5, 4.2, -31.0)),
        CameraMode::HeroStreet => (
            Point3::new(1.35, 2.85, 20.0),
            Point3::new(-0.25, 1.85, -16.0),
        ),
        CameraMode::CinematicOrbit => {
            let angle = time * 0.055;
            let eye = Point3::new(
                angle.sin() * 24.0,
                10.5 + (time * 0.2).sin() * 0.8,
                angle.cos() * 24.0,
            );
            (eye, Point3::new(0.0, 2.8 + (time * 0.13).cos() * 0.5, 0.0))
        }
    }
}

const MAT_ASPHALT: f32 = 0.0;
const MAT_CONCRETE: f32 = 1.0;
const MAT_BRICK: f32 = 2.0;
const MAT_WINDOW_PANE: f32 = 3.0;
const MAT_CURTAIN_WALL: f32 = 4.0;
const MAT_EMISSIVE_WINDOW: f32 = 6.0;
const MAT_METAL: f32 = 7.0;
#[allow(dead_code)]
const MAT_MARKING: f32 = 8.0;
#[allow(dead_code)]
const MAT_FOLIAGE: f32 = 9.0;
const MAT_ROOF_TAR: f32 = 10.0;
const MAT_SOLAR: f32 = 11.0;

fn build_city_mesh() -> Mesh {
    load_city_gltf_mesh()
}

#[derive(Default)]
struct App {
    state: Rc<RefCell<Option<State>>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.borrow().is_some() {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        let attrs = Window::default_attributes().with_title("Rust wgpu 3D City");
        #[cfg(target_arch = "wasm32")]
        let attrs = Window::default_attributes()
            .with_title("Rust wgpu 3D City")
            .with_append(true);
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        #[cfg(target_arch = "wasm32")]
        sync_canvas_to_viewport(&window);

        #[cfg(not(target_arch = "wasm32"))]
        {
            *self.state.borrow_mut() = Some(pollster::block_on(State::new(window)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let state = Rc::clone(&self.state);
            wasm_bindgen_futures::spawn_local(async move {
                *state.borrow_mut() = Some(State::new(window).await);
                if let Some(state) = state.borrow().as_ref() {
                    state.window.request_redraw();
                }
            });
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        *self.state.borrow_mut() = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let mut state_ref = self.state.borrow_mut();
        let Some(state) = state_ref.as_mut() else {
            return;
        };
        if window_id != state.window.id() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::CursorMoved { position, .. } => state.handle_cursor_moved(position),
            WindowEvent::MouseInput {
                state: input_state,
                button,
                ..
            } => {
                state.handle_mouse_input(button, input_state);
            }
            WindowEvent::MouseWheel { delta, .. } => state.handle_mouse_wheel(delta),
            WindowEvent::RedrawRequested => {
                #[cfg(target_arch = "wasm32")]
                {
                    let size = sync_canvas_to_viewport(&state.window);
                    if size.width != state.config.width || size.height != state.config.height {
                        state.resize(size.width, size.height);
                    }
                }

                match state.render() {
                    RenderOutcome::Presented => {}
                    RenderOutcome::Lost | RenderOutcome::Outdated => {
                        state.resize(state.config.width, state.config.height)
                    }
                    RenderOutcome::Timeout | RenderOutcome::Occluded => {}
                    RenderOutcome::Validation => event_loop.exit(),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.borrow().as_ref() {
            state.window.request_redraw();
        }
    }
}

fn run_app() {
    let event_loop = EventLoop::new().expect("create event loop");
    let app = App::default();

    #[cfg(target_arch = "wasm32")]
    event_loop.spawn_app(app);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("run app");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
#[cfg(target_arch = "wasm32")]
pub fn wasm_main() {
    console_error_panic_hook::set_once();
    run_app();
}

#[cfg(target_arch = "wasm32")]
pub fn main() {}

#[cfg(not(target_arch = "wasm32"))]
pub fn main() {
    run_app();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gltf_material_names_map_to_shader_materials() {
        assert_eq!(gltf_material("asphalt").1, MAT_ASPHALT);
        assert_eq!(gltf_material("curtain_wall").1, MAT_CURTAIN_WALL);
        assert_eq!(gltf_material("emissive_window").1, MAT_EMISSIVE_WINDOW);
        assert_eq!(gltf_material("unknown").1, MAT_CONCRETE);
    }
}
