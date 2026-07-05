struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    horizon_color: vec4<f32>,
    ambient_color: vec4<f32>,
    settings: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var hdr_scene: texture_2d<f32>;
@group(1) @binding(1)
var hdr_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material_id: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material_id: f32,
};

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn material_response(id: f32) -> vec4<f32> {
    if (id < 0.5) { return vec4<f32>(0.72, 0.10, 0.00, 0.0); }      // asphalt rough, wet sheen
    if (id < 1.5) { return vec4<f32>(0.88, 0.04, 0.00, 0.0); }      // concrete
    if (id < 2.5) { return vec4<f32>(0.78, 0.05, 0.00, 0.0); }      // brick
    if (id < 3.5) { return vec4<f32>(0.28, 0.65, 0.00, 0.0); }      // glass
    if (id < 4.5) { return vec4<f32>(0.38, 0.18, 3.20, 1.0); }      // lit window/sign emissive
    if (id < 5.5) { return vec4<f32>(0.46, 0.28, 0.00, 0.0); }      // metal roof
    if (id < 6.5) { return vec4<f32>(0.80, 0.02, 0.00, 0.0); }      // road marking
    return vec4<f32>(0.70, 0.03, 0.00, 0.0);                       // foliage
}

fn sky_color_for_dir(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    var sky = mix(uniforms.horizon_color.rgb, uniforms.sky_color.rgb, pow(up, 0.72));
    let sun_dot = max(dot(normalize(dir), normalize(uniforms.sun_direction.xyz)), 0.0);
    let glow = pow(sun_dot, 360.0) * 8.0 + pow(sun_dot, 24.0) * 0.85;
    sky += uniforms.sun_color.rgb * glow;
    return sky;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.world_position = input.position;
    out.color = input.color;
    out.normal = normalize(input.normal);
    out.uv = input.uv;
    out.material_id = input.material_id;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);
    let v = normalize(uniforms.camera_position.xyz - input.world_position);
    let sun = normalize(uniforms.sun_direction.xyz);
    let m = material_response(input.material_id);
    var base = input.color;

    let facade_noise = hash21(floor(input.world_position.xz * 2.2) + floor(input.uv * 7.0));
    base *= 0.88 + facade_noise * 0.22;
    let grime = smoothstep(5.5, 0.0, input.world_position.y) * (0.08 + 0.12 * hash21(input.world_position.xz));
    if (input.material_id > 0.5 && input.material_id < 3.5) { base *= 1.0 - grime; }

    if (input.material_id > 2.5 && input.material_id < 3.5) {
        let cell = floor(input.uv * vec2<f32>(7.0, 18.0));
        let local = fract(input.uv * vec2<f32>(7.0, 18.0));
        let frame = step(0.08, local.x) * step(local.x, 0.92) * step(0.10, local.y) * step(local.y, 0.78);
        let lit = step(0.57, hash21(cell + floor(input.world_position.xz)));
        let warm = mix(vec3<f32>(0.55, 0.72, 1.0), vec3<f32>(1.0, 0.62, 0.25), hash21(cell + 9.1));
        base = mix(base * vec3<f32>(0.08, 0.13, 0.18), warm, frame * lit * 0.85);
    }

    let diffuse = max(dot(n, sun), 0.0);
    let half_vec = normalize(sun + v);
    let spec_power = mix(18.0, 96.0, m.y);
    let spec = pow(max(dot(n, half_vec), 0.0), spec_power) * m.y;
    let fresnel = pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let reflection = sky_color_for_dir(reflect(-v, n)) * (m.y * 0.32 + fresnel * m.y);
    let ambient = uniforms.ambient_color.rgb * (0.55 + 0.45 * max(n.y, 0.0));
    var color = base * (ambient + uniforms.sun_color.rgb * diffuse * (1.25 - m.x * 0.28)) + reflection + uniforms.sun_color.rgb * spec;

    if (m.w > 0.5) { color += base * m.z; }
    if (input.material_id < 0.5) {
        let lane_reflect = pow(max(dot(reflect(-sun, n), v), 0.0), 26.0);
        color += uniforms.sun_color.rgb * lane_reflect * 0.38;
        color *= 0.82 + 0.18 * hash21(input.world_position.xz * 8.0);
    }

    let dist = length(input.world_position - uniforms.camera_position.xyz);
    let height_fog = clamp(1.0 - input.world_position.y / 18.0, 0.15, 1.0);
    let fog = 1.0 - exp(-dist * uniforms.settings.z * height_fog);
    let view_dir = normalize(input.world_position - uniforms.camera_position.xyz);
    color = mix(color, sky_color_for_dir(view_dir), clamp(fog, 0.0, 0.82));
    return vec4<f32>(color, 1.0);
}

struct FullOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> FullOut {
    var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(3.0, 1.0), vec2<f32>(-1.0, 1.0));
    var out: FullOut;
    out.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    out.uv = pos[vertex_index] * 0.5 + vec2<f32>(0.5);
    return out;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_post(input: FullOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_scene));
    let texel = 1.0 / dims;
    var color = textureSample(hdr_scene, hdr_sampler, input.uv).rgb;
    var bloom = vec3<f32>(0.0);
    for (var x: i32 = -2; x <= 2; x = x + 1) {
        for (var y: i32 = -2; y <= 2; y = y + 1) {
            let s = textureSample(hdr_scene, hdr_sampler, input.uv + vec2<f32>(f32(x), f32(y)) * texel * 2.0).rgb;
            bloom += max(s - vec3<f32>(1.15), vec3<f32>(0.0)) / 25.0;
        }
    }
    color += bloom * 0.22 * uniforms.settings.w;
    color *= uniforms.settings.y;
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = mix(color * vec3<f32>(0.88, 0.94, 1.06), color * vec3<f32>(1.08, 0.98, 0.88), smoothstep(0.35, 1.8, luma));
    let d = distance(input.uv, vec2<f32>(0.5));
    color *= 1.0 - smoothstep(0.42, 0.82, d) * 0.20;
    color += (hash21(input.uv * dims + uniforms.settings.xx) - 0.5) * 0.012;
    color = aces_tonemap(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(color, 1.0);
}
