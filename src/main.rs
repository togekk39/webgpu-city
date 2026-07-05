use cgmath::{Deg, Matrix4, Point3, SquareMatrix, Vector3, perspective};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{cell::RefCell, rc::Rc, sync::Arc};
use wgpu::util::DeviceExt;
#[cfg(target_arch = "wasm32")]
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
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
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    light_dir: [f32; 4],
    time_effects: [f32; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            light_dir: [-0.82, 0.28, 0.42, 0.0],
            time_effects: [0.0, 1.0, 1.0, 1.0],
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

    fn add_box(&mut self, center: [f32; 3], size: [f32; 3], color: [f32; 3]) {
        let [cx, cy, cz] = center;
        let [sx, sy, sz] = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
        let base = self.vertices.len() as u32;
        let corners = [
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
            [cx + sx, cy - sy, cz - sz],
            [cx - sx, cy - sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz - sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz + sz],
            [cx - sx, cy - sy, cz + sz],
        ];
        self.vertices.extend(
            corners
                .into_iter()
                .map(|position| Vertex { position, color }),
        );
        for face in 0..6 {
            let o = base + face * 4;
            self.indices
                .extend_from_slice(&[o, o + 1, o + 2, o, o + 2, o + 3]);
        }
    }
}

struct DepthTexture {
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

impl DepthTexture {
    fn create(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let format = wgpu::TextureFormat::Depth24Plus;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("city depth texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
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
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    draw_calls: u32,
    triangle_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth: DepthTexture,
    clock: AppClock,
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

        let mesh = build_city_mesh();
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
        uniforms.view_proj = camera_matrix(config.width, config.height, 0.0).into();
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
        let depth = DepthTexture::create(&device, &config);
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("city pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout)],
            immediate_size: 0,
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
                    format,
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
        Self {
            window,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices: mesh.indices.len() as u32,
            draw_calls: 1,
            triangle_count: mesh.indices.len() as u32 / 3,
            uniform_buffer,
            uniform_bind_group,
            depth,
            clock: AppClock::new(),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth = DepthTexture::create(&self.device, &self.config);
    }

    fn update(&self) {
        let elapsed = self.clock.elapsed_secs();
        let uniforms = Uniforms {
            view_proj: camera_matrix(self.config.width, self.config.height, elapsed * 0.035).into(),
            light_dir: [-0.82, 0.28, 0.42, 0.0],
            // x=time, y=fog, z=bloom/emissive lift, w=color grade toggle.
            time_effects: [elapsed, 1.0, 1.0, 1.0],
        };
        if (elapsed * 2.0) as u32 % 2 == 0 {
            self.window.set_title(&format!(
                "Rust wgpu Hero Block | {:.0} tris | {} draw | fog/bloom/colorgrade on",
                self.triangle_count, self.draw_calls
            ));
        }
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
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
                label: Some("city render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.05,
                            b: 0.09,
                            a: 1.0,
                        }),
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
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        RenderOutcome::Presented
    }
}

fn camera_matrix(width: u32, height: u32, angle: f32) -> Matrix4<f32> {
    let aspect = width as f32 / height as f32;
    let eye = Point3::new(-18.0 + angle.sin() * 3.0, 14.0, 31.0 + angle.cos() * 2.0);
    let view = Matrix4::look_at_rh(eye, Point3::new(1.5, 3.0, -18.0), Vector3::unit_y());
    let proj = perspective(Deg(52.0), aspect, 0.1, 160.0);
    proj * view
}

fn hash01(seed: i32) -> f32 {
    let n = (seed as f32 * 12.9898).sin() * 43_758.547;
    n.fract().abs()
}

fn add_facade_details(mesh: &mut Mesh, cx: f32, cz: f32, sx: f32, height: f32, style: i32) {
    let floors = (height / (1.05 + hash01(style) * 0.28)).floor().max(3.0) as i32;
    let warm = [[1.0, 0.66, 0.30], [0.85, 0.48, 0.24], [0.55, 0.74, 0.95]];
    let dark = [0.018, 0.026, 0.034];
    let window_w = if style % 3 == 0 { 0.42 } else { 0.30 };
    for floor in 0..floors {
        let y = 0.85 + floor as f32 * (height - 1.4) / floors as f32;
        let slots = (sx / (0.82 + hash01(style + floor) * 0.45))
            .floor()
            .max(3.0) as i32;
        for slot in 0..slots {
            let pattern = (slot * 13 + floor * 7 + style * 5).rem_euclid(11);
            if pattern == 0 || pattern == 6 {
                continue;
            }
            let lit = matches!(pattern, 1 | 3 | 8) || (floor < 2 && slot % 3 == 0);
            let glass = if lit {
                warm[(style as usize + slot as usize) % warm.len()]
            } else {
                dark
            };
            let x = cx + ((slot as f32 + 0.5) / slots as f32 - 0.5) * sx * 0.78;
            mesh.add_box([x, y, cz + 2.03], [window_w, 0.46, 0.045], glass);
            if floor % 4 == 1 && slot % 2 == 0 {
                mesh.add_box(
                    [x, y - 0.34, cz + 2.08],
                    [window_w + 0.22, 0.05, 0.22],
                    [0.16, 0.17, 0.16],
                );
            }
        }
        if floor % 3 == 0 {
            mesh.add_box(
                [cx, y + 0.36, cz + 2.055],
                [sx * 0.93, 0.04, 0.05],
                [0.28, 0.25, 0.22],
            );
        }
    }
    if style % 2 == 0 {
        mesh.add_box(
            [cx - sx * 0.42, height * 0.5, cz + 2.08],
            [0.06, height * 0.82, 0.06],
            [0.08, 0.085, 0.08],
        );
    }
    mesh.add_box(
        [cx, height + 0.08, cz],
        [sx * 1.04, 0.16, 4.15],
        [0.20, 0.20, 0.19],
    );
}

fn build_hero_building(
    mesh: &mut Mesh,
    cx: f32,
    cz: f32,
    sx: f32,
    sz: f32,
    height: f32,
    style: i32,
    shop: bool,
) {
    let palette = [
        [0.31, 0.16, 0.10],
        [0.22, 0.20, 0.18],
        [0.38, 0.32, 0.25],
        [0.18, 0.22, 0.26],
        [0.28, 0.24, 0.20],
        [0.13, 0.14, 0.15],
        [0.44, 0.27, 0.17],
        [0.24, 0.28, 0.29],
        [0.34, 0.34, 0.31],
        [0.20, 0.17, 0.14],
    ];
    let mut body = palette[style as usize % palette.len()];
    let tint = hash01(style * 19) * 0.08 - 0.035;
    body = [body[0] + tint, body[1] + tint * 0.6, body[2] + tint * 0.3];
    mesh.add_box([cx, height * 0.5, cz], [sx, height, sz], body);
    add_facade_details(mesh, cx, cz + sz * 0.5 - 2.0, sx, height, style);
    if shop {
        mesh.add_box(
            [cx, 0.62, cz + sz * 0.52],
            [sx * 0.86, 1.05, 0.10],
            [0.055, 0.04, 0.035],
        );
        mesh.add_box(
            [cx, 1.34, cz + sz * 0.55],
            [sx * 0.72, 0.30, 0.14],
            [0.95, 0.31 + hash01(style) * 0.25, 0.10],
        );
        mesh.add_box(
            [cx - sx * 0.22, 0.62, cz + sz * 0.59],
            [0.72, 0.78, 0.08],
            [0.95, 0.62, 0.28],
        );
        mesh.add_box(
            [cx + sx * 0.25, 0.62, cz + sz * 0.59],
            [0.72, 0.78, 0.08],
            [0.05, 0.08, 0.09],
        );
        mesh.add_box(
            [cx, 1.78, cz + sz * 0.59],
            [sx * 0.78, 0.14, 0.46],
            [0.32, 0.05, 0.04],
        );
    }
    // grime bands, leaks, AC boxes, pipes
    mesh.add_box(
        [cx, 0.12, cz + sz * 0.525],
        [sx * 0.92, 0.22, 0.05],
        [0.055, 0.050, 0.044],
    );
    for i in 0..3 {
        let x = cx + (hash01(style * 31 + i) - 0.5) * sx * 0.74;
        mesh.add_box(
            [x, height * (0.32 + i as f32 * 0.14), cz + sz * 0.535],
            [0.08, height * 0.18, 0.04],
            [0.06, 0.052, 0.045],
        );
    }
    mesh.add_box(
        [cx + sx * 0.32, height + 0.36, cz],
        [0.56, 0.44, 0.72],
        [0.25, 0.28, 0.28],
    );
}

fn build_city_mesh() -> Mesh {
    let mut mesh = Mesh::new();
    mesh.add_box(
        [0.0, -0.08, 0.0],
        [100.0, 0.16, 100.0],
        [0.045, 0.052, 0.055],
    );
    // main wet avenue, side street, alley, sidewalks and curbs
    mesh.add_box([0.0, 0.02, 0.0], [13.0, 0.05, 100.0], [0.018, 0.019, 0.020]);
    mesh.add_box(
        [18.0, 0.025, -10.0],
        [7.0, 0.05, 72.0],
        [0.017, 0.018, 0.020],
    );
    mesh.add_box(
        [0.0, 0.03, -16.0],
        [100.0, 0.05, 8.0],
        [0.016, 0.017, 0.018],
    );
    for x in [-8.0, 8.0, 14.2, 21.8] {
        mesh.add_box([x, 0.08, 0.0], [0.35, 0.18, 100.0], [0.34, 0.30, 0.24]);
    }
    for x in [-11.5, 11.5, 25.5] {
        mesh.add_box([x, 0.05, 0.0], [6.5, 0.08, 100.0], [0.12, 0.115, 0.105]);
    }
    for z in [-20.4, -11.6] {
        mesh.add_box([0.0, 0.08, z], [100.0, 0.18, 0.35], [0.34, 0.30, 0.24]);
    }
    for lane in [-2.7, 2.7] {
        for z in (-48..48).step_by(9) {
            mesh.add_box(
                [lane, 0.075, z as f32],
                [0.16, 0.035, 3.4],
                [0.80, 0.70, 0.42],
            );
        }
    }
    for z in (-44..42).step_by(12) {
        mesh.add_box(
            [-5.1, 0.07, z as f32],
            [3.2, 0.035, 0.18],
            [0.82, 0.80, 0.72],
        );
    }

    let blocks = [
        (-15.0, -34.0, 7.0, 10.0, 12.0),
        (-16.0, -21.0, 8.5, 8.0, 18.0),
        (-16.5, -6.0, 9.0, 9.5, 9.0),
        (-15.5, 8.0, 8.0, 12.0, 15.0),
        (-15.0, 25.0, 9.0, 10.0, 22.0),
        (14.0, -36.0, 8.5, 10.0, 16.0),
        (14.5, -23.0, 7.5, 7.5, 11.0),
        (30.0, -20.0, 10.0, 13.0, 24.0),
        (13.5, 2.0, 8.0, 11.0, 14.0),
        (15.0, 20.0, 9.5, 13.0, 28.0),
    ];
    for (i, b) in blocks.iter().enumerate() {
        build_hero_building(
            &mut mesh,
            b.0,
            b.1,
            b.2,
            b.3,
            b.4,
            i as i32,
            i % 2 == 0 || i == 7,
        );
    }
    // distant skyline layers / landmark focal point
    for i in 0..18 {
        let x = -42.0 + i as f32 * 5.1;
        let h = 18.0 + hash01(i * 9) * 32.0;
        build_hero_building(
            &mut mesh,
            x,
            -54.0,
            3.8 + hash01(i) * 2.5,
            5.0,
            h,
            30 + i,
            false,
        );
    }
    build_hero_building(&mut mesh, 35.0, -58.0, 8.0, 7.0, 48.0, 77, false);
    mesh.add_box([35.0, 73.5, -58.0], [0.20, 3.5, 0.20], [1.0, 0.72, 0.35]);
    // street lights, poles, wires, signs, cars, planters, bins
    for (i, z) in (-42..40).step_by(10).enumerate() {
        for x in [-7.2, 7.2, 13.3, 22.7] {
            mesh.add_box([x, 1.8, z as f32], [0.10, 3.5, 0.10], [0.08, 0.08, 0.075]);
            mesh.add_box(
                [x + if x < 0.0 { -0.45 } else { 0.45 }, 3.45, z as f32],
                [0.9, 0.08, 0.08],
                [0.09, 0.085, 0.075],
            );
            mesh.add_box(
                [x + if x < 0.0 { -0.86 } else { 0.86 }, 3.35, z as f32],
                [0.22, 0.22, 0.22],
                [1.0, 0.55, 0.20],
            );
        }
        mesh.add_box(
            [0.0, 0.36, z as f32 + hash01(i as i32) * 3.0],
            [1.65, 0.42, 3.0],
            [0.05 + hash01(i as i32) * 0.2, 0.06, 0.07],
        );
        mesh.add_box(
            [0.0, 0.85, z as f32 + 1.2],
            [1.25, 0.18, 1.6],
            [0.95, 0.82, 0.55],
        );
    }
    for i in 0..14 {
        let x = if i % 2 == 0 { -10.2 } else { 10.4 };
        let z = -43.0 + i as f32 * 6.3;
        mesh.add_box([x, 0.35, z], [0.55, 0.70, 0.55], [0.08, 0.15, 0.08]);
        mesh.add_box([x, 0.9, z], [0.95, 0.30, 0.95], [0.05, 0.18, 0.06]);
    }
    for i in 0..10 {
        mesh.add_box(
            [-6.2, 2.8, -44.0 + i as f32 * 8.0],
            [0.04, 0.04, 8.0],
            [0.035, 0.032, 0.03],
        );
    }
    mesh
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
