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
    sky_color: [f32; 4],
    horizon_color: [f32; 4],
    ambient_color: [f32; 4],
    settings: [f32; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            inv_view_proj: Matrix4::identity().into(),
            camera_position: [0.0, 0.0, 0.0, 1.0],
            sun_direction: [-0.92, 0.16, -0.36, 0.0],
            sun_color: [1.0, 0.55, 0.20, 1.0],
            sky_color: [0.08, 0.13, 0.23, 1.0],
            horizon_color: [1.0, 0.42, 0.12, 1.0],
            ambient_color: [0.12, 0.17, 0.26, 1.0],
            settings: [0.0, 1.08, 0.032, 1.0],
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
    render_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    post_layout: wgpu::BindGroupLayout,
    hdr: HdrTarget,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth: DepthTexture,
    render_scale: f32,
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
        let render_scale = configured_render_scale();
        let hdr_width = scaled_extent(config.width, render_scale);
        let hdr_height = scaled_extent(config.height, render_scale);

        let mesh = build_city_mesh();
        eprintln!(
            "city stats: vertices={} indices={} draw_calls=3 render_scale={:.2} postprocess=bloom+vignette+filmic",
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
        let (initial_view_proj, initial_eye) = camera_matrix(config.width, config.height, 0.0);
        uniforms.view_proj = initial_view_proj.into();
        uniforms.inv_view_proj = initial_view_proj
            .invert()
            .unwrap_or_else(Matrix4::identity)
            .into();
        uniforms.camera_position = [initial_eye.x, initial_eye.y, initial_eye.z, 1.0];
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
            bind_group_layouts: &[Some(&uniform_layout)],
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
            bind_group_layouts: &[Some(&uniform_layout), Some(&post_layout)],
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
            render_pipeline,
            post_pipeline,
            post_layout,
            hdr,
            vertex_buffer,
            index_buffer,
            num_indices: mesh.indices.len() as u32,
            uniform_buffer,
            uniform_bind_group,
            depth,
            render_scale,
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
        let hdr_width = scaled_extent(self.config.width, self.render_scale);
        let hdr_height = scaled_extent(self.config.height, self.render_scale);
        self.depth = DepthTexture::create(&self.device, hdr_width, hdr_height);
        self.hdr = HdrTarget::create(&self.device, hdr_width, hdr_height, &self.post_layout);
    }

    fn update(&self) {
        let elapsed = self.clock.elapsed_secs();
        let (view_proj, eye) = camera_matrix(self.config.width, self.config.height, elapsed);
        let mut uniforms = Uniforms::new();
        uniforms.view_proj = view_proj.into();
        uniforms.inv_view_proj = view_proj.invert().unwrap_or_else(Matrix4::identity).into();
        uniforms.camera_position = [eye.x, eye.y, eye.z, 1.0];
        uniforms.settings[0] = elapsed;
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
            pass.set_bind_group(1, &self.hdr.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        RenderOutcome::Presented
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
    CinematicOrbit,
    HeroStreet,
}

fn camera_matrix(width: u32, height: u32, time: f32) -> (Matrix4<f32>, Point3<f32>) {
    let aspect = width as f32 / height as f32;
    let mode = if option_env!("WEBGPU_CITY_CAMERA") == Some("orbit") {
        CameraMode::CinematicOrbit
    } else {
        CameraMode::HeroStreet
    };
    let (eye, target) = match mode {
        CameraMode::HeroStreet => (Point3::new(5.8, 7.0, 18.5), Point3::new(-0.55, 2.2, -11.0)),
        CameraMode::CinematicOrbit => {
            let angle = time * 0.055;
            let eye = Point3::new(
                angle.sin() * 24.0,
                10.5 + (time * 0.2).sin() * 0.8,
                angle.cos() * 24.0,
            );
            (eye, Point3::new(0.0, 2.8 + (time * 0.13).cos() * 0.5, 0.0))
        }
    };
    let view = Matrix4::look_at_rh(eye, target, Vector3::unit_y());
    let proj = perspective(Deg(50.0), aspect, 0.1, 140.0);
    (proj * view, eye)
}

const MAT_ASPHALT: f32 = 0.0;
const MAT_CONCRETE: f32 = 1.0;
const MAT_BRICK: f32 = 2.0;
const MAT_GLASS: f32 = 3.0;
const MAT_EMISSIVE: f32 = 4.0;
const MAT_METAL: f32 = 5.0;
const MAT_MARKING: f32 = 6.0;
const MAT_FOLIAGE: f32 = 7.0;

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
        FacadeKind::GlassOffice => ([0.07, 0.13 + wobble, 0.18 + wobble], MAT_GLASS),
        FacadeKind::MixedUse => ([0.30 + wobble, 0.26 + wobble, 0.21 + wobble], MAT_CONCRETE),
        FacadeKind::LowShop => ([0.45 + wobble, 0.36 + wobble, 0.26 + wobble], MAT_CONCRETE),
        FacadeKind::Corner => ([0.22 + wobble, 0.16 + wobble, 0.14 + wobble], MAT_BRICK),
        FacadeKind::Tower => ([0.10, 0.16 + wobble, 0.25 + wobble], MAT_GLASS),
    }
}

fn build_building(mesh: &mut Mesh, center: [f32; 3], size: [f32; 3], kind: FacadeKind, seed: i32) {
    let [cx, _, cz] = center;
    let [sx, height, sz] = size;
    let (color, mat) = facade_color(kind, seed);
    mesh.add_box([cx, height * 0.5, cz], [sx, height, sz], color, mat);

    // Setbacks and roof silhouettes keep the skyline irregular without per-window cubes.
    if height > 5.0 {
        mesh.add_box(
            [cx + sx * 0.08, height + 0.45, cz - sz * 0.06],
            [sx * 0.58, 0.9, sz * 0.55],
            color,
            mat,
        );
    }
    mesh.add_box(
        [cx, height + 0.08, cz],
        [sx * 1.05, 0.16, sz * 1.05],
        [0.22, 0.24, 0.23],
        MAT_METAL,
    );
    mesh.add_box(
        [cx - sx * 0.25, height + 0.34, cz + sz * 0.22],
        [0.16, 0.36, 0.16],
        [0.42, 0.45, 0.44],
        MAT_METAL,
    );
    if seed.rem_euclid(3) == 0 {
        mesh.add_box(
            [cx + sx * 0.24, height + 0.25, cz - sz * 0.18],
            [0.42, 0.20, 0.30],
            [0.31, 0.36, 0.38],
            MAT_METAL,
        );
    }
    if matches!(
        kind,
        FacadeKind::MixedUse | FacadeKind::LowShop | FacadeKind::Corner
    ) {
        mesh.add_box(
            [cx, 0.75, cz + sz * 0.515],
            [sx * 0.72, 0.42, 0.035],
            [1.0, 0.46, 0.16],
            MAT_EMISSIVE,
        );
        mesh.add_box(
            [cx, 1.10, cz + sz * 0.525],
            [sx * 0.82, 0.08, 0.16],
            [0.18, 0.05, 0.035],
            MAT_METAL,
        );
    }
}

fn build_streetlight(mesh: &mut Mesh, x: f32, z: f32) {
    mesh.add_box(
        [x, 0.8, z],
        [0.06, 1.6, 0.06],
        [0.18, 0.16, 0.14],
        MAT_METAL,
    );
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
        MAT_EMISSIVE,
    );
}

fn build_city_mesh() -> Mesh {
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
        mesh.add_box(
            [x, 0.04, 0.0],
            [1.15, 0.08, 42.0],
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
        mesh.add_box([x, 0.20, z], [0.58, 0.28, 1.05], c, MAT_METAL);
        mesh.add_box(
            [x, 0.43, z - 0.05],
            [0.42, 0.22, 0.52],
            [0.04, 0.06, 0.08],
            MAT_GLASS,
        );
    }
    for x in [-3.35, 3.35] {
        for z in (-18..=18).step_by(6) {
            mesh.add_box(
                [x, 0.55, z as f32],
                [0.55, 1.1, 0.55],
                [0.08, 0.20, 0.10],
                MAT_FOLIAGE,
            );
        }
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
