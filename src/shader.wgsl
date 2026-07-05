struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    light_view_proj: mat4x4<f32>,
    sky_color: vec4<f32>,
    horizon_color: vec4<f32>,
    ambient_color: vec4<f32>,
    settings: vec4<f32>,
    sunset: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var shadow_map: texture_depth_2d;
@group(1) @binding(1)
var shadow_sampler: sampler_comparison;

@group(2) @binding(0)
var hdr_scene: texture_2d<f32>;
@group(2) @binding(1)
var hdr_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material_id: f32,
    @location(5) shadow_position: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material_id: f32,
    @location(5) shadow_position: vec4<f32>,
};

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        v += value_noise(q) * amp;
        q *= 2.03;
        amp *= 0.5;
    }
    return v;
}

fn material_response(id: f32) -> vec4<f32> {
    if (id < 0.5) { return vec4<f32>(0.78, 0.08, 0.00, 0.0); }      // asphalt rough, faintly patched
    if (id < 1.5) { return vec4<f32>(0.88, 0.04, 0.00, 0.0); }      // concrete
    if (id < 2.5) { return vec4<f32>(0.80, 0.04, 0.00, 0.0); }      // brick/stucco
    if (id < 3.5) { return vec4<f32>(0.18, 0.48, 0.00, 0.0); }      // single geometry window pane: no nested grid
    if (id < 4.5) { return vec4<f32>(0.24, 0.72, 0.00, 0.0); }      // curtain wall: procedural facade grid allowed
    if (id < 5.5) { return vec4<f32>(0.16, 0.78, 0.00, 0.0); }      // shop glass
    if (id < 6.5) { return vec4<f32>(0.42, 0.14, 1.65, 1.0); }      // emissive windows/signs, controlled bloom
    if (id < 7.5) { return vec4<f32>(0.48, 0.24, 0.00, 0.0); }      // metal
    if (id < 8.5) { return vec4<f32>(0.82, 0.02, 0.00, 0.0); }      // road marking
    if (id < 9.5) { return vec4<f32>(0.70, 0.03, 0.00, 0.0); }      // foliage
    if (id < 10.5) { return vec4<f32>(0.92, 0.06, 0.00, 0.0); }     // dark roof tar atlas tile
    return vec4<f32>(0.22, 0.62, 0.00, 0.0);                       // solar/gloss panel atlas tile
}

fn sky_color_for_dir(dir: vec3<f32>) -> vec3<f32> {
    let d = normalize(dir);
    let sun = normalize(uniforms.sun_direction.xyz);
    let mu = clamp(dot(d, sun), -1.0, 1.0);
    let up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
    let haze = uniforms.sunset.w;
    let zenith = uniforms.sky_color.rgb * (0.86 + 0.22 * (1.0 - haze * 0.2));
    let horizon = uniforms.horizon_color.rgb * (0.62 + haze * 0.22);
    var sky = mix(horizon, zenith, pow(up, 0.62));
    let horizon_glow = exp(-max(d.y, 0.0) * 7.5) * (0.45 + haze * 0.20);
    sky += uniforms.horizon_color.rgb * horizon_glow;
    let forward = pow(max(mu, 0.0), mix(18.0, 8.0, clamp(haze - 0.8, 0.0, 1.0))) * (0.55 + haze * 0.45);
    let corona = pow(max(mu, 0.0), 90.0) * 2.8;
    let disc = smoothstep(cos(uniforms.sunset.z * 1.15), cos(uniforms.sunset.z * 0.72), mu);
    sky += uniforms.sun_color.rgb * (forward + corona);
    sky += uniforms.sun_color.rgb * disc * 24.0;
    return sky * 1.12;
}

fn sample_shadow(world_pos: vec3<f32>, normal: vec3<f32>, sun: vec3<f32>) -> f32 {
    let n_dot_l = clamp(dot(normal, sun), 0.0, 1.0);
    let normal_bias = normal * (0.035 + (1.0 - n_dot_l) * 0.075);
    let lp = uniforms.light_view_proj * vec4<f32>(world_pos + normal_bias, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    let receiver_bias = 0.0012 + (1.0 - n_dot_l) * 0.0025;
    var lit = 0.0;
    for (var x: i32 = -1; x <= 1; x = x + 1) {
        for (var y: i32 = -1; y <= 1; y = y + 1) {
            lit += textureSampleCompare(shadow_map, shadow_sampler, uv + vec2<f32>(f32(x), f32(y)) * texel * 1.7, ndc.z - receiver_bias);
        }
    }
    return lit / 9.0;
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
    out.shadow_position = uniforms.light_view_proj * vec4<f32>(input.position, 1.0);
    return out;
}

@vertex
fn vs_shadow(input: VertexInput) -> @builtin(position) vec4<f32> {
    return uniforms.light_view_proj * vec4<f32>(input.position, 1.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);
    let v = normalize(uniforms.camera_position.xyz - input.world_position);
    let sun = normalize(uniforms.sun_direction.xyz);
    let m = material_response(input.material_id);
    var base = input.color;

    let fine_noise = fbm(input.world_position.xz * 0.65 + input.uv * 0.35);
    base *= 0.985 + fine_noise * 0.015;
    let atlas_uv = fract(input.uv * 0.55);
    if (input.material_id > 1.5 && input.material_id < 2.5) {
        let mortar_x = step(0.07, atlas_uv.x) * step(atlas_uv.x, 0.93);
        let mortar_y = step(0.13, atlas_uv.y) * step(atlas_uv.y, 0.87);
        base *= mix(vec3<f32>(0.82,0.78,0.72), vec3<f32>(1.02,0.94,0.86), mortar_x * mortar_y);
    }
    if (input.material_id > 0.5 && input.material_id < 1.5) {
        let aggregate = smoothstep(0.25, 0.95, fbm(input.uv * 3.7));
        base *= mix(vec3<f32>(0.78,0.78,0.76), vec3<f32>(1.10,1.07,1.00), aggregate * 0.45);
    }
    if (input.material_id > 9.5 && input.material_id < 10.5) {
        let seams = min(smoothstep(0.035, 0.055, fract(input.uv.x * 1.7)), smoothstep(0.035, 0.055, fract(input.uv.y * 1.2)));
        base *= mix(vec3<f32>(0.58,0.56,0.54), vec3<f32>(1.0), seams);
    }
    if (input.material_id > 10.5 && input.material_id < 11.5) {
        let grid = step(0.08, fract(input.uv.x * 6.0)) * step(0.08, fract(input.uv.y * 2.0));
        base = mix(vec3<f32>(0.015,0.020,0.025), base * 1.25, grid);
    }
    let rain = fbm(vec2<f32>(input.world_position.x * 1.8 + input.world_position.z * 0.25, input.world_position.y * 0.13));
    let streaks = smoothstep(0.48, 0.86, rain) * smoothstep(14.0, 1.0, input.world_position.y);
    let grime = streaks * 0.10;
    if (input.material_id > 0.5 && input.material_id < 5.5) { base *= 1.0 - grime; }

    if (input.material_id > 3.5 && input.material_id < 4.5) {
        let cell = floor(input.uv * vec2<f32>(7.0, 18.0));
        let local = fract(input.uv * vec2<f32>(7.0, 18.0));
        let frame = step(0.08, local.x) * step(local.x, 0.92) * step(0.10, local.y) * step(local.y, 0.78);
        let lit = step(0.57, hash21(cell + floor(input.world_position.xz)));
        let warm = mix(vec3<f32>(0.55, 0.72, 1.0), vec3<f32>(1.0, 0.62, 0.25), hash21(cell + 9.1));
        base = mix(base * vec3<f32>(0.05, 0.08, 0.11), warm * 0.42, frame * lit * 0.30);
    }

    if (input.material_id > 2.5 && input.material_id < 3.5) {
        base = mix(base, vec3<f32>(0.015, 0.030, 0.045), 0.55);
        base += vec3<f32>(0.10, 0.045, 0.018) * step(0.82, hash21(floor(input.world_position.xy * 2.0)));
    }
    if (input.material_id > 4.5 && input.material_id < 5.5) {
        base = mix(base, vec3<f32>(0.018, 0.024, 0.028), 0.45);
        base += vec3<f32>(0.18, 0.09, 0.035) * smoothstep(0.68, 0.95, hash21(floor(input.uv * vec2<f32>(5.0, 2.0))));
    }

    let diffuse = max(dot(n, sun), 0.0);
    let shadow = sample_shadow(input.world_position, n, sun);
    let dist = length(input.world_position - uniforms.camera_position.xyz);
    let shadow_influence = mix(0.92, 0.35, smoothstep(45.0, 95.0, dist));
    let direct_shadow = mix(1.0 - shadow_influence, 1.0, shadow);
    let half_vec = normalize(sun + v);
    let spec_power = mix(22.0, 128.0, m.y);
    let spec = pow(max(dot(n, half_vec), 0.0), spec_power) * m.y * direct_shadow;
    let fresnel = pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let reflection = sky_color_for_dir(reflect(-v, n)) * (m.y * 0.22 + fresnel * m.y * 0.75);
    let sky_ambient = uniforms.ambient_color.rgb * (0.48 + 0.52 * max(n.y, 0.0));
    let cool_bounce = vec3<f32>(0.045, 0.060, 0.115) * (1.0 - diffuse) * (0.65 + 0.35 * max(n.y, 0.0));
    let direct = uniforms.sun_color.rgb * diffuse * direct_shadow * (1.48 - m.x * 0.34);
    var color = base * (sky_ambient + cool_bounce + direct) + reflection + uniforms.sun_color.rgb * spec;

    if (m.w > 0.5) { color += base * m.z * 0.58; }
    if (input.material_id < 0.5) {
        let lane_reflect = pow(max(dot(reflect(-sun, n), v), 0.0), 26.0);
        color += uniforms.sun_color.rgb * lane_reflect * 0.55 * direct_shadow;
        color *= 0.82 + 0.18 * hash21(input.world_position.xz * 8.0);
    }

    let height_fog = clamp(1.15 - input.world_position.y / 18.0, 0.12, 1.0);
    let depth_bias = smoothstep(20.0, 105.0, dist);
    let view_dir = normalize(input.world_position - uniforms.camera_position.xyz);
    let sun_view = pow(max(dot(view_dir, sun), 0.0), 8.0);
    let fog = (1.0 - exp(-dist * uniforms.settings.z * 0.55 * height_fog)) * depth_bias;
    let warm_haze = mix(sky_color_for_dir(view_dir), uniforms.horizon_color.rgb * 0.55 + uniforms.sun_color.rgb * (0.25 + sun_view * 0.55), 0.45);
    color = mix(color, warm_haze, clamp(fog, 0.0, 0.72));
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
fn fs_sky(input: FullOut) -> @location(0) vec4<f32> {
    let ndc = vec4<f32>(input.uv * 2.0 - vec2<f32>(1.0), 1.0, 1.0);
    let world = uniforms.inv_view_proj * ndc;
    let dir = normalize(world.xyz / world.w - uniforms.camera_position.xyz);
    var color = sky_color_for_dir(dir);
    let cloud_band = smoothstep(0.02, 0.22, dir.y) * (1.0 - smoothstep(0.42, 0.82, dir.y));
    let cloud_p = (dir.xz / max(dir.y + 0.18, 0.06)) * 2.6 + uniforms.settings.xx * 0.012;
    let cloud_noise = fbm(cloud_p);
    let cloud = cloud_band * smoothstep(0.48, 0.76, cloud_noise) * 0.16;
    color = mix(color, uniforms.sun_color.rgb * 0.85 + vec3<f32>(0.35, 0.19, 0.14), cloud);
    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_post(input: FullOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_scene));
    let texel = 1.0 / dims;
    let scene_uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
    var color = textureSample(hdr_scene, hdr_sampler, scene_uv).rgb;
    var bloom = vec3<f32>(0.0);
    for (var x: i32 = -2; x <= 2; x = x + 1) {
        for (var y: i32 = -2; y <= 2; y = y + 1) {
            let s = textureSample(hdr_scene, hdr_sampler, scene_uv + vec2<f32>(f32(x), f32(y)) * texel * 2.0).rgb;
            bloom += max(s - vec3<f32>(1.15), vec3<f32>(0.0)) / 25.0;
        }
    }
    color += bloom * 0.16 * uniforms.settings.w;
    color *= uniforms.settings.y * 1.03;
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = mix(color * vec3<f32>(0.88, 0.94, 1.06), color * vec3<f32>(1.08, 0.98, 0.88), smoothstep(0.35, 1.8, luma));
    let d = distance(input.uv, vec2<f32>(0.5));
    color *= 1.0 - smoothstep(0.52, 0.92, d) * 0.045;
    color = aces_tonemap(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(color, 1.0);
}
