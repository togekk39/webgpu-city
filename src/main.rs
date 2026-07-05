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
    event::{ElementState, MouseButton, WindowEvent},
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
            sunset: [5.5_f32.to_radians(), 220.0_f32.to_radians(), 0.0105, 1.25],
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

    fn add_beveled_box(
        &mut self,
        center: [f32; 3],
        size: [f32; 3],
        bevel: f32,
        color: [f32; 3],
        material_id: f32,
    ) {
        let [cx, cy, cz] = center;
        let [hx, hy, hz] = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
        let b = bevel.min(hx * 0.35).min(hy * 0.35).min(hz * 0.35).max(0.0);
        if b <= 0.001 {
            self.add_box(center, size, color, material_id);
            return;
        }
        let xs = [cx - hx, cx - hx + b, cx + hx - b, cx + hx];
        let ys = [cy - hy, cy - hy + b, cy + hy - b, cy + hy];
        let zs = [cz - hz, cz - hz + b, cz + hz - b, cz + hz];
        let mut quad = |pts: [[f32; 3]; 4], normal: [f32; 3], uv: [f32; 2]| {
            let base = self.vertices.len() as u32;
            let uvs = [[0.0, 0.0], [uv[0], 0.0], [uv[0], uv[1]], [0.0, uv[1]]];
            for i in 0..4 {
                self.vertices.push(Vertex {
                    position: pts[i],
                    color,
                    normal,
                    uv: uvs[i],
                    material_id,
                });
            }
            self.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        };
        quad(
            [
                [xs[1], ys[0], zs[3]],
                [xs[2], ys[0], zs[3]],
                [xs[2], ys[3], zs[3]],
                [xs[1], ys[3], zs[3]],
            ],
            [0.0, 0.0, 1.0],
            [size[0], size[1]],
        );
        quad(
            [
                [xs[2], ys[0], zs[0]],
                [xs[1], ys[0], zs[0]],
                [xs[1], ys[3], zs[0]],
                [xs[2], ys[3], zs[0]],
            ],
            [0.0, 0.0, -1.0],
            [size[0], size[1]],
        );
        quad(
            [
                [xs[0], ys[0], zs[1]],
                [xs[0], ys[0], zs[2]],
                [xs[0], ys[3], zs[2]],
                [xs[0], ys[3], zs[1]],
            ],
            [-1.0, 0.0, 0.0],
            [size[2], size[1]],
        );
        quad(
            [
                [xs[3], ys[0], zs[2]],
                [xs[3], ys[0], zs[1]],
                [xs[3], ys[3], zs[1]],
                [xs[3], ys[3], zs[2]],
            ],
            [1.0, 0.0, 0.0],
            [size[2], size[1]],
        );
        quad(
            [
                [xs[1], ys[3], zs[3]],
                [xs[2], ys[3], zs[3]],
                [xs[2], ys[3], zs[0]],
                [xs[1], ys[3], zs[0]],
            ],
            [0.0, 1.0, 0.0],
            [size[0], size[2]],
        );
        quad(
            [
                [xs[1], ys[0], zs[0]],
                [xs[2], ys[0], zs[0]],
                [xs[2], ys[0], zs[3]],
                [xs[1], ys[0], zs[3]],
            ],
            [0.0, -1.0, 0.0],
            [size[0], size[2]],
        );
        let n = 0.70710677;
        for &(z0, z1, nz) in &[(zs[2], zs[3], n), (zs[0], zs[1], -n)] {
            quad(
                [
                    [xs[0], ys[1], z1],
                    [xs[1], ys[0], z0],
                    [xs[1], ys[3], z0],
                    [xs[0], ys[2], z1],
                ],
                [-n, 0.0, nz],
                [b, size[1]],
            );
            quad(
                [
                    [xs[2], ys[0], z0],
                    [xs[3], ys[1], z1],
                    [xs[3], ys[2], z1],
                    [xs[2], ys[3], z0],
                ],
                [n, 0.0, nz],
                [b, size[1]],
            );
        }
        for &(y0, y1, ny) in &[(ys[2], ys[3], n), (ys[0], ys[1], -n)] {
            quad(
                [
                    [xs[1], y1, zs[3]],
                    [xs[2], y1, zs[3]],
                    [xs[2], y0, zs[2]],
                    [xs[1], y0, zs[2]],
                ],
                [0.0, ny, n],
                [size[0], b],
            );
            quad(
                [
                    [xs[2], y1, zs[0]],
                    [xs[1], y1, zs[0]],
                    [xs[1], y0, zs[1]],
                    [xs[2], y0, zs[1]],
                ],
                [0.0, ny, -n],
                [size[0], b],
            );
            quad(
                [
                    [xs[0], y1, zs[1]],
                    [xs[0], y1, zs[2]],
                    [xs[1], y0, zs[2]],
                    [xs[1], y0, zs[1]],
                ],
                [-n, ny, 0.0],
                [size[2], b],
            );
            quad(
                [
                    [xs[3], y1, zs[2]],
                    [xs[3], y1, zs[1]],
                    [xs[2], y0, zs[1]],
                    [xs[2], y0, zs[2]],
                ],
                [n, ny, 0.0],
                [size[2], b],
            );
        }
    }

    fn add_prism(
        &mut self,
        center: [f32; 3],
        radius: f32,
        height: f32,
        sides: u32,
        color: [f32; 3],
        material_id: f32,
    ) {
        let n = sides.max(6);
        let [cx, cy, cz] = center;
        for i in 0..n {
            let a0 = i as f32 / n as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
            let (x0, z0) = (cx + a0.cos() * radius, cz + a0.sin() * radius);
            let (x1, z1) = (cx + a1.cos() * radius, cz + a1.sin() * radius);
            let mid = (a0 + a1) * 0.5;
            let normal = Vector3::new(mid.cos(), 0.0, mid.sin()).normalize();
            let base = self.vertices.len() as u32;
            for p in [
                [x0, cy - height * 0.5, z0],
                [x1, cy - height * 0.5, z1],
                [x1, cy + height * 0.5, z1],
                [x0, cy + height * 0.5, z0],
            ] {
                self.vertices.push(Vertex {
                    position: p,
                    color,
                    normal: normal.into(),
                    uv: [0.0, 0.0],
                    material_id,
                });
            }
            self.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    fn add_box(&mut self, center: [f32; 3], size: [f32; 3], color: [f32; 3], material_id: f32) {
        let [cx, cy, cz] = center;
        let [hx, hy, hz] = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
        let faces = [
            (
                [
                    [cx - hx, cy - hy, cz + hz],
                    [cx + hx, cy - hy, cz + hz],
                    [cx + hx, cy + hy, cz + hz],
                    [cx - hx, cy + hy, cz + hz],
                ],
                [0.0, 0.0, 1.0],
                [size[0], size[1]],
            ),
            (
                [
                    [cx + hx, cy - hy, cz - hz],
                    [cx - hx, cy - hy, cz - hz],
                    [cx - hx, cy + hy, cz - hz],
                    [cx + hx, cy + hy, cz - hz],
                ],
                [0.0, 0.0, -1.0],
                [size[0], size[1]],
            ),
            (
                [
                    [cx - hx, cy - hy, cz - hz],
                    [cx - hx, cy - hy, cz + hz],
                    [cx - hx, cy + hy, cz + hz],
                    [cx - hx, cy + hy, cz - hz],
                ],
                [-1.0, 0.0, 0.0],
                [size[2], size[1]],
            ),
            (
                [
                    [cx + hx, cy - hy, cz + hz],
                    [cx + hx, cy - hy, cz - hz],
                    [cx + hx, cy + hy, cz - hz],
                    [cx + hx, cy + hy, cz + hz],
                ],
                [1.0, 0.0, 0.0],
                [size[2], size[1]],
            ),
            (
                [
                    [cx - hx, cy + hy, cz + hz],
                    [cx + hx, cy + hy, cz + hz],
                    [cx + hx, cy + hy, cz - hz],
                    [cx - hx, cy + hy, cz - hz],
                ],
                [0.0, 1.0, 0.0],
                [size[0], size[2]],
            ),
            (
                [
                    [cx - hx, cy - hy, cz - hz],
                    [cx + hx, cy - hy, cz - hz],
                    [cx + hx, cy - hy, cz + hz],
                    [cx - hx, cy - hy, cz + hz],
                ],
                [0.0, -1.0, 0.0],
                [size[0], size[2]],
            ),
        ];
        for (corners, normal, uv_scale) in faces {
            let base = self.vertices.len() as u32;
            let uvs = [
                [0.0, 0.0],
                [uv_scale[0], 0.0],
                [uv_scale[0], uv_scale[1]],
                [0.0, uv_scale[1]],
            ];
            for i in 0..4 {
                self.vertices.push(Vertex {
                    position: corners[i],
                    color,
                    normal,
                    uv: uvs[i],
                    material_id,
                });
            }
            self.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
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
        let elapsed = self.clock.elapsed_secs();
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

    fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        self.camera.handle_cursor_moved(position);
    }

    fn handle_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        self.camera.handle_mouse_input(button, state);
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
    last_cursor: Option<PhysicalPosition<f64>>,
    left_dragging: bool,
}

impl CameraController {
    const ROTATION_SENSITIVITY: f32 = 0.003;
    const PAN_SENSITIVITY: f32 = 0.018;
    const MIN_CAMERA_Y: f32 = 0.25;
    const MIN_PITCH: f32 = -0.35;
    const MAX_PITCH: f32 = 0.85;

    fn new() -> Self {
        Self {
            yaw_offset: 0.0,
            pitch_offset: 0.0,
            pan: Vector3::new(0.0, 0.0, 0.0),
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
        let radius = offset.magnitude();
        let base_yaw = offset.x.atan2(offset.z);
        let base_pitch = (offset.y / radius).asin();
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
    let elevation = 5.5_f32.to_radians();
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
const MAT_SHOP_GLASS: f32 = 5.0;
const MAT_EMISSIVE_WINDOW: f32 = 6.0;
const MAT_METAL: f32 = 7.0;
#[allow(dead_code)]
const MAT_MARKING: f32 = 8.0;
#[allow(dead_code)]
const MAT_FOLIAGE: f32 = 9.0;
const MAT_ROOF_TAR: f32 = 10.0;
const MAT_SOLAR: f32 = 11.0;

#[derive(Clone, Copy)]
enum FacadeKind {
    OldApartment,
    GlassOffice,
    MixedUse,
    LowShop,
    Corner,
    Tower,
}

fn facade_color(kind: FacadeKind, seed: i32) -> ([f32; 3], f32) {
    let wobble = ((seed * 37).rem_euclid(17) as f32) * 0.006;
    match kind {
        FacadeKind::OldApartment => ([0.34 + wobble, 0.18 + wobble, 0.12 + wobble], MAT_BRICK),
        FacadeKind::GlassOffice => ([0.07, 0.13 + wobble, 0.18 + wobble], MAT_CURTAIN_WALL),
        FacadeKind::MixedUse => ([0.30 + wobble, 0.26 + wobble, 0.21 + wobble], MAT_CONCRETE),
        FacadeKind::LowShop => ([0.45 + wobble, 0.36 + wobble, 0.26 + wobble], MAT_CONCRETE),
        FacadeKind::Corner => ([0.22 + wobble, 0.16 + wobble, 0.14 + wobble], MAT_BRICK),
        FacadeKind::Tower => ([0.10, 0.16 + wobble, 0.25 + wobble], MAT_CURTAIN_WALL),
    }
}

fn window_lit(seed: i32, bay: i32, floor: i32) -> bool {
    let floor_cluster = (seed + floor * 17).rem_euclid(11) < 3;
    let dark_side = (seed + bay * 5).rem_euclid(13) < 2;
    let resident = (seed * 31 + bay * 19 + floor * 23).rem_euclid(17) == 0;
    (floor_cluster && !dark_side && (bay + seed).rem_euclid(3) != 0) || resident
}

fn add_panel_window(mesh: &mut Mesh, pos: [f32; 3], size: [f32; 3], lit: bool) {
    mesh.add_box(
        pos,
        [size[0] + 0.07, size[1] + 0.07, size[2]],
        [0.025, 0.027, 0.030],
        MAT_METAL,
    );
    mesh.add_box(
        [pos[0], pos[1], pos[2] + size[2] * 0.35],
        size,
        if lit {
            [0.78, 0.42, 0.18]
        } else {
            [0.025, 0.045, 0.062]
        },
        if lit {
            MAT_EMISSIVE_WINDOW
        } else {
            MAT_WINDOW_PANE
        },
    );
}

fn build_old_apartment(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], seed: i32) {
    let [cx, _, cz] = center;
    let [sx, h, sz] = size;
    let side = if cx > 0.0 { -1.0 } else { 1.0 };
    let fz = cz + side * sz * 0.5;
    let color = facade_color(FacadeKind::OldApartment, seed).0;
    mesh.add_beveled_box([cx, h * 0.5, cz], [sx, h, sz], 0.035, color, MAT_BRICK);
    let floors = ((h - 1.1) / 0.78).max(2.0) as i32;
    let bays = (sx / 0.48).max(3.0) as i32;
    for f in 0..floors {
        let y = 1.05 + f as f32 * (0.72 + ((seed + f) % 3) as f32 * 0.035);
        if y > h - 0.35 {
            continue;
        };
        mesh.add_box(
            [cx, y - 0.34, fz + side * 0.035],
            [sx * 0.92, 0.035, 0.05],
            [0.16, 0.12, 0.10],
            MAT_CONCRETE,
        );
        for b in 0..bays {
            let x = cx - sx * 0.38 + (b as f32 + 0.5) * sx * 0.76 / bays as f32;
            add_panel_window(
                mesh,
                [x, y, fz + side * 0.075],
                [sx * 0.40 / bays as f32, 0.28, 0.035],
                window_lit(seed, b, f),
            );
            if (seed + b * 3 + f).rem_euclid(7) == 0 {
                mesh.add_beveled_box(
                    [x, y - 0.25, fz + side * 0.22],
                    [sx * 0.45 / bays as f32, 0.045, 0.32],
                    0.01,
                    [0.13, 0.13, 0.12],
                    MAT_METAL,
                );
            }
            if (seed + b * 5 + f).rem_euclid(13) == 0 {
                mesh.add_box(
                    [x + 0.18, y - 0.08, fz + side * 0.18],
                    [0.18, 0.12, 0.10],
                    [0.36, 0.38, 0.36],
                    MAT_METAL,
                );
            }
        }
    }
    mesh.add_beveled_box(
        [cx, h + 0.10, cz],
        [sx * 1.04, 0.20, sz * 1.03],
        0.025,
        [0.18, 0.18, 0.17],
        MAT_METAL,
    );
}

fn build_glass_office(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], seed: i32) {
    let [cx, _, cz] = center;
    let [sx, h, sz] = size;
    let side = if cx > 0.0 { -1.0 } else { 1.0 };
    let fz = cz + side * sz * 0.5;
    mesh.add_beveled_box(
        [cx, 0.75, cz],
        [sx * 1.08, 1.5, sz * 1.08],
        0.04,
        [0.10, 0.11, 0.12],
        MAT_CONCRETE,
    );
    mesh.add_beveled_box(
        [cx, h * 0.52, cz],
        [sx * 0.82, h - 1.2, sz * 0.86],
        0.025,
        [0.035, 0.075, 0.105],
        MAT_CURTAIN_WALL,
    );
    mesh.add_box(
        [cx, h * 0.52, fz + side * 0.05],
        [sx * 0.76, h - 1.6, 0.05],
        [0.035, 0.095, 0.13],
        MAT_CURTAIN_WALL,
    );
    for b in 0..5 {
        let x = cx - sx * 0.31 + b as f32 * sx * 0.155;
        mesh.add_box(
            [x, h * 0.52, fz + side * 0.085],
            [0.035, h - 1.4, 0.06],
            [0.16, 0.18, 0.18],
            MAT_METAL,
        );
    }
    for f in 0..((h / 0.9) as i32) {
        let y = 1.5 + f as f32 * 0.9;
        mesh.add_box(
            [cx, y, fz + side * 0.09],
            [sx * 0.78, 0.035, 0.065],
            [0.18, 0.19, 0.18],
            MAT_METAL,
        );
    }
    mesh.add_beveled_box(
        [cx, h + 0.28, cz],
        [sx * 0.62, 0.55, sz * 0.66],
        0.035,
        [0.06, 0.09, 0.12],
        MAT_CURTAIN_WALL,
    );
    if seed % 2 == 0 {
        mesh.add_box(
            [cx + sx * 0.18, h * 0.62, cz - side * sz * 0.18],
            [sx * 0.38, h * 0.42, sz * 0.18],
            [0.05, 0.08, 0.10],
            MAT_CURTAIN_WALL,
        );
    }
}

fn build_mixed_use(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], seed: i32) {
    let [cx, _, cz] = center;
    let [sx, h, sz] = size;
    let side = if cx > 0.0 { -1.0 } else { 1.0 };
    let fz = cz + side * sz * 0.5;
    mesh.add_beveled_box(
        [cx, h * 0.5, cz],
        [sx, h, sz],
        0.035,
        facade_color(FacadeKind::MixedUse, seed).0,
        MAT_CONCRETE,
    );
    mesh.add_beveled_box(
        [cx, 0.85, fz + side * 0.05],
        [sx * 0.92, 1.35, 0.08],
        0.015,
        [0.035, 0.040, 0.045],
        MAT_SHOP_GLASS,
    );
    let sign = [
        0.25 + 0.08 * (seed % 3) as f32,
        0.08 + 0.05 * ((seed + 1) % 3) as f32,
        0.04,
    ];
    mesh.add_box(
        [cx, 1.62, fz + side * 0.10],
        [sx * (0.45 + 0.08 * (seed % 4) as f32), 0.26, 0.07],
        sign,
        MAT_EMISSIVE_WINDOW,
    );
    mesh.add_box(
        [cx, 1.33, fz + side * 0.24],
        [sx * 0.75, 0.07, 0.38],
        [0.12, 0.06, 0.04],
        MAT_METAL,
    );
    for f in 0..((h - 2.0) / 0.82).max(1.0) as i32 {
        for b in 0..(sx / 0.55).max(2.0) as i32 {
            let x = cx - sx * 0.32 + (b as f32 + 0.5) * sx * 0.64 / (sx / 0.55).max(2.0).floor();
            add_panel_window(
                mesh,
                [x, 2.15 + f as f32 * 0.82, fz + side * 0.08],
                [0.28, 0.25, 0.035],
                window_lit(seed, b, f),
            );
        }
    }
}

fn build_low_shop(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], seed: i32) {
    let mut s = size;
    s[1] = s[1].min(2.4);
    build_mixed_use(mesh, center, s, seed);
    let [cx, _, cz] = center;
    mesh.add_beveled_box(
        [cx, s[1] + 0.25, cz],
        [s[0] * 0.72, 0.30, s[2] * 0.55],
        0.03,
        [0.22, 0.24, 0.24],
        MAT_METAL,
    );
}
fn build_corner(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], seed: i32) {
    let [cx, _, cz] = center;
    let [sx, h, sz] = size;
    build_old_apartment(mesh, center, size, seed);
    let side = if cx > 0.0 { -1.0 } else { 1.0 };
    mesh.add_beveled_box(
        [cx - side * sx * 0.38, h * 0.45, cz + sz * 0.38],
        [sx * 0.35, h * 0.82, sz * 0.35],
        0.04,
        [0.19, 0.12, 0.10],
        MAT_BRICK,
    );
    mesh.add_box(
        [cx - side * sx * 0.46, 1.2, cz + sz * 0.48],
        [0.08, 1.7, 0.9],
        [0.025, 0.04, 0.05],
        MAT_SHOP_GLASS,
    );
}
fn build_tower(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], seed: i32) {
    let [cx, _, cz] = center;
    let [sx, h, sz] = size;
    build_glass_office(mesh, [cx, 0.0, cz], [sx * 1.2, h * 0.45, sz * 1.2], seed);
    mesh.add_beveled_box(
        [cx, h * 0.62, cz],
        [sx * 0.62, h * 0.78, sz * 0.62],
        0.03,
        [0.04, 0.08, 0.13],
        MAT_CURTAIN_WALL,
    );
    mesh.add_beveled_box(
        [cx, h + 0.45, cz],
        [sx * 0.45, 0.9, sz * 0.45],
        0.025,
        [0.08, 0.10, 0.13],
        MAT_METAL,
    );
    mesh.add_prism(
        [cx, h + 1.15, cz],
        0.035,
        1.1,
        8,
        [0.7, 0.45, 0.22],
        MAT_EMISSIVE_WINDOW,
    );
}

fn build_building(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], kind: FacadeKind, seed: i32) {
    match kind {
        FacadeKind::OldApartment => build_old_apartment(mesh, center, size, seed),
        FacadeKind::GlassOffice => build_glass_office(mesh, center, size, seed),
        FacadeKind::MixedUse => build_mixed_use(mesh, center, size, seed),
        FacadeKind::LowShop => build_low_shop(mesh, center, size, seed),
        FacadeKind::Corner => build_corner(mesh, center, size, seed),
        FacadeKind::Tower => build_tower(mesh, center, size, seed),
    }
}

#[allow(dead_code)]
fn build_streetlight(mesh: &mut Mesh, x: f32, z: f32) {
    mesh.add_prism([x, 0.8, z], 0.04, 1.6, 8, [0.18, 0.16, 0.14], MAT_METAL);
    mesh.add_box(
        [x, 1.62, z - 0.18],
        [0.08, 0.08, 0.34],
        [0.18, 0.16, 0.14],
        MAT_METAL,
    );
    mesh.add_box(
        [x, 1.58, z - 0.38],
        [0.16, 0.10, 0.10],
        [1.0, 0.62, 0.28],
        MAT_EMISSIVE_WINDOW,
    );
}

fn add_rooftop_kit(mesh: &mut Mesh, cx: f32, roof_y: f32, cz: f32, sx: f32, sz: f32, seed: i32) {
    let parapet = [0.11, 0.105, 0.095];
    mesh.add_box(
        [cx, roof_y + 0.22, cz - sz * 0.50],
        [sx + 0.12, 0.44, 0.12],
        parapet,
        MAT_CONCRETE,
    );
    mesh.add_box(
        [cx, roof_y + 0.22, cz + sz * 0.50],
        [sx + 0.12, 0.44, 0.12],
        parapet,
        MAT_CONCRETE,
    );
    mesh.add_box(
        [cx - sx * 0.50, roof_y + 0.22, cz],
        [0.12, 0.44, sz],
        parapet,
        MAT_CONCRETE,
    );
    mesh.add_box(
        [cx + sx * 0.50, roof_y + 0.22, cz],
        [0.12, 0.44, sz],
        parapet,
        MAT_CONCRETE,
    );
    mesh.add_beveled_box(
        [cx - sx * 0.23, roof_y + 0.18, cz + sz * 0.12],
        [sx * 0.22, 0.20, sz * 0.28],
        0.015,
        [0.030, 0.028, 0.026],
        MAT_ROOF_TAR,
    );
    mesh.add_beveled_box(
        [cx + sx * 0.18, roof_y + 0.32, cz - sz * 0.18],
        [sx * 0.20, 0.64, sz * 0.18],
        0.025,
        [0.22, 0.22, 0.20],
        MAT_METAL,
    );
    mesh.add_prism(
        [cx - sx * 0.34, roof_y + 0.48, cz - sz * 0.28],
        0.20,
        0.42,
        12,
        [0.30, 0.31, 0.30],
        MAT_METAL,
    );
    mesh.add_prism(
        [cx - sx * 0.34, roof_y + 0.90, cz - sz * 0.28],
        0.13,
        0.44,
        12,
        [0.08, 0.07, 0.06],
        MAT_METAL,
    );
    mesh.add_beveled_box(
        [cx + sx * 0.32, roof_y + 0.62, cz + sz * 0.26],
        [0.50, 0.92, 0.42],
        0.02,
        [0.18, 0.16, 0.14],
        MAT_BRICK,
    );
    mesh.add_box(
        [cx + sx * 0.05, roof_y + 0.18, cz + sz * 0.32],
        [sx * 0.24, 0.12, 0.18],
        [0.09, 0.17, 0.22],
        MAT_SOLAR,
    );
    mesh.add_box(
        [cx + sx * 0.05, roof_y + 0.24, cz + sz * 0.32],
        [sx * 0.24, 0.025, 0.19],
        [0.025, 0.035, 0.045],
        MAT_WINDOW_PANE,
    );
    for i in 0..3 {
        let x = cx - sx * 0.30 + i as f32 * sx * 0.22;
        mesh.add_prism(
            [x, roof_y + 0.32, cz + sz * 0.02],
            0.065,
            0.42,
            8,
            [0.20, 0.20, 0.19],
            MAT_METAL,
        );
        mesh.add_box(
            [x, roof_y + 0.56, cz + sz * 0.02],
            [0.24, 0.10, 0.24],
            [0.18, 0.18, 0.17],
            MAT_METAL,
        );
    }
    if seed % 2 == 0 {
        for i in 0..5 {
            mesh.add_box(
                [
                    cx - sx * 0.40 + i as f32 * sx * 0.20,
                    roof_y + 0.52,
                    cz + sz * 0.48,
                ],
                [0.08, 0.08, 0.08],
                [1.0, 0.55, 0.22],
                MAT_EMISSIVE_WINDOW,
            );
        }
    }
}

fn build_rooftop_foreground(mesh: &mut Mesh) {
    let specs = [
        (-8.2, 7.6, 10.0, 4.6, 4.2, 4.0),
        (-2.8, 6.4, 12.6, 5.6, 3.7, 3.1),
        (3.9, 7.2, 10.8, 5.2, 4.6, 3.7),
        (9.3, 6.6, 8.4, 4.3, 4.0, 3.3),
        (-11.0, 5.8, 4.6, 3.7, 4.8, 3.0),
        (-5.0, 5.2, 5.6, 5.0, 4.5, 2.8),
        (1.6, 5.9, 5.1, 4.8, 5.0, 3.2),
        (7.7, 5.3, 3.2, 4.6, 4.1, 2.6),
        (-9.2, 4.9, -0.4, 4.1, 4.2, 2.5),
        (-3.0, 4.5, -0.9, 4.8, 3.8, 2.4),
        (3.3, 4.8, -1.7, 5.0, 4.5, 2.9),
        (9.6, 4.2, -2.8, 4.5, 4.0, 2.4),
    ];
    for (i, (x, h, z, sx, sz, base)) in specs.iter().enumerate() {
        let mat = if i % 3 == 0 { MAT_BRICK } else { MAT_CONCRETE };
        let col = if mat == MAT_BRICK {
            [0.25, 0.12, 0.08]
        } else {
            [0.28, 0.26, 0.23]
        };
        mesh.add_beveled_box([*x, h * 0.5, *z], [*sx, *h, *sz], 0.035, col, mat);
        mesh.add_box(
            [*x, *h + 0.035, *z],
            [*sx * 0.94, 0.07, *sz * 0.94],
            [0.035, 0.032, 0.030],
            MAT_ROOF_TAR,
        );
        add_rooftop_kit(mesh, *x, *h + 0.08, *z, *sx * 0.86, *sz * 0.86, i as i32);
        let bays = 3 + (i as i32 % 3);
        for b in 0..bays {
            for f in 0..2 {
                if (i as i32 + b + f) % 3 == 0 {
                    add_panel_window(
                        mesh,
                        [
                            *x - *sx * 0.34 + b as f32 * *sx * 0.22,
                            base + f as f32 * 0.75,
                            *z + *sz * 0.51,
                        ],
                        [0.28, 0.24, 0.035],
                        (i + b as usize) % 4 == 0,
                    );
                }
            }
        }
    }
}

fn build_midground_blocks(mesh: &mut Mesh) {
    let kinds = [
        FacadeKind::OldApartment,
        FacadeKind::MixedUse,
        FacadeKind::GlassOffice,
        FacadeKind::LowShop,
        FacadeKind::Corner,
        FacadeKind::Tower,
    ];
    for row in 0..5 {
        for col in 0..9 {
            let seed = row * 37 + col * 19;
            let x = -21.0 + col as f32 * 5.2 + ((seed % 5) as f32 - 2.0) * 0.45;
            let z = -8.0 - row as f32 * 7.2 + ((seed % 7) as f32 - 3.0) * 0.35;
            let sx = 2.5 + (seed % 4) as f32 * 0.55;
            let sz = 2.3 + ((seed / 3) % 4) as f32 * 0.50;
            let h = 2.6 + (seed % 9) as f32 * 0.70 + if col % 5 == 0 { 2.8 } else { 0.0 };
            build_building(
                mesh,
                [x, 0.0, z],
                [sx, h, sz],
                kinds[(seed as usize) % kinds.len()],
                seed as i32,
            );
            mesh.add_box(
                [x, h + 0.04, z],
                [sx * 0.86, 0.06, sz * 0.86],
                [0.04, 0.038, 0.035],
                MAT_ROOF_TAR,
            );
            if row < 3 {
                add_rooftop_kit(mesh, x, h + 0.05, z, sx * 0.72, sz * 0.72, seed as i32);
            }
        }
    }
    for i in 0..11 {
        let x = -24.0 + i as f32 * 4.8;
        mesh.add_box(
            [x, 0.015, -20.0 - (i % 3) as f32 * 5.5],
            [3.0, 0.03, 18.0],
            [0.018, 0.018, 0.017],
            MAT_ASPHALT,
        );
    }
}

fn build_background_skyline(mesh: &mut Mesh) {
    for i in 0..44 {
        let x = -44.0 + i as f32 * 2.05;
        let h = 5.5
            + ((i * 13) % 17) as f32 * 0.75
            + if i == 34 {
                13.0
            } else if i == 22 {
                7.0
            } else {
                0.0
            };
        let w = 1.0 + ((i * 7) % 5) as f32 * 0.28;
        let z = -56.0 - ((i * 5) % 9) as f32 * 1.5;
        let mat = if i % 4 == 0 {
            MAT_CURTAIN_WALL
        } else {
            MAT_CONCRETE
        };
        let col = if mat == MAT_CURTAIN_WALL {
            [0.035, 0.060, 0.085]
        } else {
            [0.12, 0.10, 0.095]
        };
        mesh.add_box([x, h * 0.5, z], [w, h, 1.1], col, mat);
        if i == 34 {
            mesh.add_prism(
                [x, h + 2.2, z],
                0.05,
                4.2,
                8,
                [0.75, 0.50, 0.25],
                MAT_EMISSIVE_WINDOW,
            );
        }
        if i == 22 {
            mesh.add_beveled_box(
                [x, h + 0.9, z],
                [w * 0.55, 1.8, 0.7],
                0.02,
                [0.08, 0.08, 0.09],
                MAT_METAL,
            );
        }
    }
}

#[allow(dead_code)]
fn build_hero_street_mesh() -> Mesh {
    let mut mesh = Mesh::new();
    mesh.add_box(
        [0.0, -0.08, 0.0],
        [42.0, 0.16, 46.0],
        [0.08, 0.09, 0.09],
        MAT_CONCRETE,
    );
    mesh.add_box(
        [0.0, 0.01, 0.0],
        [4.2, 0.05, 42.0],
        [0.018, 0.018, 0.017],
        MAT_ASPHALT,
    );
    for x in [-1.05, 1.05] {
        mesh.add_box(
            [x, 0.055, 0.0],
            [0.08, 0.035, 38.0],
            [0.82, 0.72, 0.50],
            MAT_MARKING,
        );
    }
    for z in (-18..=18).step_by(4) {
        mesh.add_box(
            [0.0, 0.06, z as f32],
            [0.14, 0.035, 1.35],
            [0.85, 0.76, 0.55],
            MAT_MARKING,
        );
    }
    for z in [-12.0, 0.0, 12.0] {
        mesh.add_box(
            [0.0, 0.07, z],
            [4.4, 0.04, 0.12],
            [0.90, 0.86, 0.76],
            MAT_MARKING,
        );
        mesh.add_box(
            [0.0, 0.01, z],
            [38.0, 0.04, 1.7],
            [0.019, 0.019, 0.018],
            MAT_ASPHALT,
        );
    }
    for x in [-3.0, 3.0] {
        mesh.add_beveled_box(
            [x, 0.04, 0.0],
            [1.15, 0.08, 42.0],
            0.025,
            [0.24, 0.23, 0.21],
            MAT_CONCRETE,
        );
    }

    let kinds = [
        FacadeKind::OldApartment,
        FacadeKind::GlassOffice,
        FacadeKind::MixedUse,
        FacadeKind::LowShop,
        FacadeKind::Corner,
        FacadeKind::Tower,
    ];
    for side in [-1.0, 1.0] {
        for i in 0..12 {
            let z = -18.0 + i as f32 * 3.3;
            let near = 1.0 - (z.abs() / 22.0).min(0.75);
            let h =
                2.2 + ((i * 7 + if side > 0.0 { 3 } else { 9 }) % 10) as f32 * 0.72 + near * 3.3;
            let w = 1.7 + (i % 3) as f32 * 0.32;
            let d = 1.8 + ((i + 1) % 3) as f32 * 0.28;
            let x = side * (4.15 + d * 0.5);
            build_building(
                &mut mesh,
                [x, 0.0, z],
                [w, h, d],
                kinds[(i as usize + if side > 0.0 { 0 } else { 2 }) % kinds.len()],
                i as i32 + if side > 0.0 { 10 } else { 40 },
            );
        }
    }
    for x in [-14.0, -10.0, -7.0, 7.0, 11.0, 15.0] {
        for z in [-14.0, -8.0, 6.0, 13.0] {
            let seed = (x as i32 * 13 + z as i32 * 7).abs();
            let h = 2.4 + (seed % 8) as f32 * 0.65;
            build_building(
                &mut mesh,
                [x, 0.0, z],
                [2.2, h, 2.0],
                kinds[seed as usize % kinds.len()],
                seed,
            );
        }
    }

    // Far end: a T-intersection, dense LOD blocks, and a lit tower anchor stop the street from falling into sky.
    mesh.add_box(
        [0.0, 0.025, -22.2],
        [22.0, 0.045, 2.2],
        [0.020, 0.020, 0.019],
        MAT_ASPHALT,
    );
    mesh.add_box(
        [0.0, 0.071, -21.3],
        [4.1, 0.035, 0.12],
        [0.88, 0.82, 0.66],
        MAT_MARKING,
    );
    for x in [-9.0, -6.6, -4.2, 4.8, 7.4, 10.2] {
        let seed = (x as i32 * 31).abs();
        build_building(
            &mut mesh,
            [x, 0.0, -24.0],
            [1.9, 3.0 + (seed % 5) as f32 * 0.55, 1.6],
            kinds[seed as usize % kinds.len()],
            seed,
        );
    }
    build_tower(&mut mesh, [2.8, 0.0, -27.0], [2.8, 11.0, 2.4], 777);
    for z in (-16..=18).step_by(4) {
        build_streetlight(&mut mesh, -2.35, z as f32);
        build_streetlight(&mut mesh, 2.35, z as f32 + 1.4);
    }
    // Simple cars and tree masses for scale; not a traffic system.
    for (x, z, c) in [
        (-0.8, -8.0, [0.08, 0.10, 0.12]),
        (0.9, -3.0, [0.55, 0.08, 0.05]),
        (-0.7, 4.0, [0.12, 0.18, 0.30]),
        (0.8, 10.0, [0.75, 0.72, 0.60]),
    ] {
        mesh.add_beveled_box([x, 0.20, z], [0.58, 0.28, 1.05], 0.06, c, MAT_METAL);
        mesh.add_box(
            [x, 0.43, z - 0.05],
            [0.42, 0.22, 0.52],
            [0.04, 0.06, 0.08],
            MAT_WINDOW_PANE,
        );
        for wx in [-0.34, 0.34] {
            for wz in [-0.34, 0.34] {
                mesh.add_prism(
                    [x + wx, 0.12, z + wz],
                    0.105,
                    0.055,
                    8,
                    [0.01, 0.01, 0.012],
                    MAT_METAL,
                );
            }
        }
    }

    // Reusable street-life kit: drainage, curb ramps, repair patches, transit furniture, utilities, and overhead cables.
    for z in [-13.0, -5.0, 7.0, 15.0] {
        mesh.add_box(
            [-2.05, 0.085, z],
            [0.34, 0.025, 0.18],
            [0.035, 0.037, 0.036],
            MAT_METAL,
        ); // drainage grate
        mesh.add_box(
            [2.05, 0.086, z + 1.2],
            [0.48, 0.024, 0.32],
            [0.045, 0.045, 0.043],
            MAT_ASPHALT,
        ); // patch
    }
    for (x, z) in [(-2.95, -12.0), (2.95, 0.0), (-2.95, 12.0)] {
        mesh.add_box(
            [x, 0.09, z],
            [0.85, 0.035, 0.55],
            [0.30, 0.28, 0.24],
            MAT_CONCRETE,
        ); // curb ramp
        mesh.add_prism(
            [x * 0.62, 0.09, z + 0.6],
            0.18,
            0.035,
            18,
            [0.055, 0.052, 0.048],
            MAT_METAL,
        ); // manhole cover
    }
    mesh.add_beveled_box(
        [-3.35, 0.68, -4.5],
        [0.12, 1.2, 1.6],
        0.015,
        [0.08, 0.12, 0.14],
        MAT_METAL,
    ); // bus-stop post
    mesh.add_box(
        [-3.25, 1.35, -4.5],
        [0.10, 0.52, 1.3],
        [0.025, 0.045, 0.06],
        MAT_WINDOW_PANE,
    ); // bus-stop glass
    mesh.add_box(
        [3.42, 0.36, 3.0],
        [0.36, 0.62, 0.28],
        [0.04, 0.13, 0.10],
        MAT_METAL,
    ); // utility box
    mesh.add_box(
        [-3.35, 0.34, 6.4],
        [0.28, 0.48, 0.28],
        [0.08, 0.09, 0.10],
        MAT_METAL,
    ); // trash can
    mesh.add_box(
        [3.28, 0.42, -9.5],
        [0.24, 0.66, 0.24],
        [0.02, 0.07, 0.16],
        MAT_METAL,
    ); // mailbox
    mesh.add_box(
        [0.0, 2.55, -2.0],
        [6.2, 0.025, 0.025],
        [0.03, 0.025, 0.022],
        MAT_METAL,
    ); // overhead cable
    mesh.add_box(
        [0.0, 2.35, 8.5],
        [5.8, 0.022, 0.022],
        [0.03, 0.025, 0.022],
        MAT_METAL,
    ); // overhead cable
    for x in [-3.35, 3.35] {
        for z in (-18..=18).step_by(6) {
            mesh.add_prism(
                [x, 0.32, z as f32],
                0.055,
                0.64,
                7,
                [0.18, 0.10, 0.05],
                MAT_BRICK,
            );
            mesh.add_box(
                [x, 0.86, z as f32],
                [0.72, 0.58, 0.035],
                [0.07, 0.22, 0.10],
                MAT_FOLIAGE,
            );
            mesh.add_box(
                [x, 0.92, z as f32],
                [0.035, 0.62, 0.72],
                [0.05, 0.18, 0.08],
                MAT_FOLIAGE,
            );
        }
    }
    mesh
}

fn build_rooftop_vista_mesh() -> Mesh {
    let mut mesh = Mesh::new();
    mesh.add_box(
        [0.0, -0.08, -18.0],
        [86.0, 0.16, 110.0],
        [0.055, 0.055, 0.052],
        MAT_CONCRETE,
    );
    build_rooftop_foreground(&mut mesh);
    build_midground_blocks(&mut mesh);
    build_background_skyline(&mut mesh);
    mesh
}

fn build_city_mesh() -> Mesh {
    match option_env!("WEBGPU_CITY_CAMERA") {
        Some("hero") | Some("street") => build_hero_street_mesh(),
        _ => build_rooftop_vista_mesh(),
    }
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
    fn city_mesh_stats_are_bounded() {
        let mesh = build_city_mesh();
        eprintln!(
            "city test stats: vertices={} indices={} draw_calls=4",
            mesh.vertices.len(),
            mesh.indices.len()
        );
        assert!(mesh.vertices.len() < 120_000);
        assert!(mesh.indices.len() < 180_000);
    }
}
