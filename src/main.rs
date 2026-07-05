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
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            light_dir: [0.45, 0.85, 0.25, 0.0],
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
            view_proj: camera_matrix(self.config.width, self.config.height, elapsed * 0.18).into(),
            light_dir: [0.45, 0.85, 0.25, 0.0],
        };
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
    let eye = Point3::new(angle.sin() * 26.0, 15.0, angle.cos() * 26.0);
    let view = Matrix4::look_at_rh(eye, Point3::new(0.0, 2.5, 0.0), Vector3::unit_y());
    let proj = perspective(Deg(45.0), aspect, 0.1, 100.0);
    proj * view
}

fn build_detailed_building(
    mesh: &mut Mesh,
    center: [f32; 3],
    size: [f32; 3],
    body_color: [f32; 3],
    accent: [f32; 3],
    seed: i32,
) {
    let [cx, _, cz] = center;
    let [sx, height, sz] = size;
    mesh.add_box([cx, height * 0.5, cz], [sx, height, sz], body_color);

    let floor_count = (height / 0.55).floor().max(2.0) as i32;
    let window_color = [0.95, 0.74, 0.32];
    let dark_glass = [0.035, 0.075, 0.12];
    let trim = [
        (body_color[0] + 0.18).min(0.75),
        (body_color[1] + 0.18).min(0.78),
        (body_color[2] + 0.18).min(0.82),
    ];

    for floor in 0..floor_count {
        let y = 0.42 + floor as f32 * 0.55;
        let lit = (floor + seed).rem_euclid(4) != 0;
        let glass = if lit { window_color } else { dark_glass };
        let window_h = 0.24;

        for side in 0..4 {
            let along = if side < 2 { sx } else { sz };
            let slots = (along / 0.45).floor().max(2.0) as i32;
            for slot in 0..slots {
                if (slot + floor + seed + side).rem_euclid(5) == 0 {
                    continue;
                }
                let offset = (slot as f32 + 0.5) / slots as f32 - 0.5;
                let span = along * offset * 0.72;
                match side {
                    0 => mesh.add_box(
                        [cx + span, y, cz + sz * 0.505],
                        [0.20, window_h, 0.035],
                        glass,
                    ),
                    1 => mesh.add_box(
                        [cx + span, y, cz - sz * 0.505],
                        [0.20, window_h, 0.035],
                        glass,
                    ),
                    2 => mesh.add_box(
                        [cx + sx * 0.505, y, cz + span],
                        [0.035, window_h, 0.20],
                        glass,
                    ),
                    _ => mesh.add_box(
                        [cx - sx * 0.505, y, cz + span],
                        [0.035, window_h, 0.20],
                        glass,
                    ),
                }
            }
        }

        if floor % 3 == 0 {
            mesh.add_box(
                [cx, y + 0.22, cz + sz * 0.512],
                [sx * 0.92, 0.035, 0.025],
                trim,
            );
            mesh.add_box(
                [cx, y + 0.22, cz - sz * 0.512],
                [sx * 0.92, 0.035, 0.025],
                trim,
            );
            mesh.add_box(
                [cx + sx * 0.512, y + 0.22, cz],
                [0.025, 0.035, sz * 0.92],
                trim,
            );
            mesh.add_box(
                [cx - sx * 0.512, y + 0.22, cz],
                [0.025, 0.035, sz * 0.92],
                trim,
            );
        }
    }

    mesh.add_box([cx, height + 0.06, cz], [sx * 1.08, 0.12, sz * 1.08], trim);
    mesh.add_box(
        [cx, height + 0.28, cz],
        [sx * 0.62, 0.32, sz * 0.58],
        accent,
    );
    mesh.add_box(
        [cx - sx * 0.22, height + 0.58, cz + sz * 0.16],
        [0.08, 0.48, 0.08],
        [0.7, 0.74, 0.72],
    );
    mesh.add_box(
        [cx + sx * 0.18, height + 0.5, cz - sz * 0.18],
        [0.26, 0.18, 0.26],
        [0.42, 0.48, 0.50],
    );
}

fn build_city_mesh() -> Mesh {
    let mut mesh = Mesh::new();
    mesh.add_box([0.0, -0.08, 0.0], [38.0, 0.16, 38.0], [0.07, 0.10, 0.12]);
    for road in [-8.0, 0.0, 8.0] {
        mesh.add_box([road, 0.01, 0.0], [1.2, 0.04, 36.0], [0.015, 0.017, 0.019]);
        mesh.add_box([0.0, 0.02, road], [36.0, 0.04, 1.2], [0.015, 0.017, 0.019]);
        mesh.add_box(
            [road - 0.62, 0.045, 0.0],
            [0.05, 0.04, 36.0],
            [0.42, 0.36, 0.22],
        );
        mesh.add_box(
            [0.0, 0.05, road + 0.62],
            [36.0, 0.04, 0.05],
            [0.42, 0.36, 0.22],
        );
    }
    for x in -5i32..=5 {
        for z in -5i32..=5 {
            if x % 3 == 0 || z % 3 == 0 {
                continue;
            }
            let xf = x as f32 * 3.0;
            let zf = z as f32 * 3.0;
            let height = 1.8 + ((x * x + z * z + 7) % 9) as f32 * 0.62;
            let color = [
                0.10 + height * 0.018,
                0.15 + (x.abs() as f32) * 0.012,
                0.22 + (z.abs() as f32) * 0.014,
            ];
            let footprint = [
                1.55 + (x.abs() % 2) as f32 * 0.28,
                height,
                1.55 + (z.abs() % 2) as f32 * 0.28,
            ];
            build_detailed_building(
                &mut mesh,
                [xf, 0.0, zf],
                footprint,
                color,
                [0.50, 0.56, 0.60],
                x * 31 + z * 17,
            );
        }
    }
    build_detailed_building(
        &mut mesh,
        [0.0, 0.0, 0.0],
        [3.0, 7.2, 3.0],
        [0.15, 0.24, 0.43],
        [0.90, 0.72, 0.34],
        91,
    );
    mesh.add_box([0.0, 7.75, 0.0], [0.16, 0.8, 0.16], [1.0, 0.86, 0.42]);
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
