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

@group(3) @binding(0)
var<uniform> material_uniform: MaterialUniform;
@group(3) @binding(1)
var base_color_texture: texture_2d<f32>;
@group(3) @binding(2)
var material_sampler: sampler;
@group(3) @binding(3)
var normal_texture: texture_2d<f32>;
@group(3) @binding(4)
var emissive_texture: texture_2d<f32>;

struct MaterialUniform {
    fallback_color: vec4<f32>,
    material: vec4<f32>,
};

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
    let zenith = uniforms.sky_color.rgb * (0.78 + 0.18 * (1.0 - haze * 0.2));
    let horizon = uniforms.horizon_color.rgb * (0.70 + haze * 0.26);
    var sky = mix(horizon, zenith, pow(up, 0.58));
    let horizon_glow = exp(-max(d.y, 0.0) * 6.2) * (0.58 + haze * 0.28);
    let dusty_band = exp(-abs(d.y) * 13.0) * (0.22 + haze * 0.18);
    sky += uniforms.horizon_color.rgb * horizon_glow;
    sky += mix(vec3<f32>(0.22, 0.18, 0.30), uniforms.sun_color.rgb, 0.55) * dusty_band;
    let forward = pow(max(mu, 0.0), mix(16.0, 6.0, clamp(haze - 0.8, 0.0, 1.0))) * (0.70 + haze * 0.52);
    let aureole = pow(max(mu, 0.0), 42.0) * 2.0;
    let hot_core = pow(max(mu, 0.0), 360.0) * 10.0;
    let soft_disc = smoothstep(cos(uniforms.sunset.z * 2.3), cos(uniforms.sunset.z * 0.45), mu);
    // Keep the sun an atmospheric glowing source rather than a hard flat disk.
    sky += uniforms.sun_color.rgb * (forward + aureole + hot_core + soft_disc * 7.0);
    return sky * 1.14;
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
    return out;
}

@vertex
fn vs_shadow(input: VertexInput) -> @builtin(position) vec4<f32> {
    return uniforms.light_view_proj * vec4<f32>(input.position, 1.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled_normal = textureSample(normal_texture, material_sampler, input.uv).rgb * 2.0 - vec3<f32>(1.0);
    let normal_mix = normalize(input.normal + sampled_normal * 0.35);
    let n = normalize(mix(input.normal, normal_mix, material_uniform.material.z));
    let v = normalize(uniforms.camera_position.xyz - input.world_position);
    let sun = normalize(uniforms.sun_direction.xyz);
    let m = material_response(material_uniform.material.x);
    let sampled_base = textureSample(base_color_texture, material_sampler, input.uv).rgb;
    var base = mix(material_uniform.fallback_color.rgb, sampled_base, material_uniform.material.y);

    let fine_noise = fbm(input.world_position.xz * 0.65 + input.uv * 0.35);
    base *= 0.985 + fine_noise * 0.015;
    let atlas_uv = fract(input.uv * 0.55);
    if (material_uniform.material.x > 1.5 && material_uniform.material.x < 2.5) {
        let mortar_x = step(0.07, atlas_uv.x) * step(atlas_uv.x, 0.93);
        let mortar_y = step(0.13, atlas_uv.y) * step(atlas_uv.y, 0.87);
        base *= mix(vec3<f32>(0.82,0.78,0.72), vec3<f32>(1.02,0.94,0.86), mortar_x * mortar_y);
    }
    if (material_uniform.material.x > 0.5 && material_uniform.material.x < 1.5) {
        let aggregate = smoothstep(0.25, 0.95, fbm(input.uv * 3.7));
        base *= mix(vec3<f32>(0.78,0.78,0.76), vec3<f32>(1.10,1.07,1.00), aggregate * 0.45);
    }
    if (material_uniform.material.x > 9.5 && material_uniform.material.x < 10.5) {
        let seams = min(smoothstep(0.035, 0.055, fract(input.uv.x * 1.7)), smoothstep(0.035, 0.055, fract(input.uv.y * 1.2)));
        base *= mix(vec3<f32>(0.58,0.56,0.54), vec3<f32>(1.0), seams);
    }
    if (material_uniform.material.x > 10.5 && material_uniform.material.x < 11.5) {
        let grid = step(0.08, fract(input.uv.x * 6.0)) * step(0.08, fract(input.uv.y * 2.0));
        base = mix(vec3<f32>(0.015,0.020,0.025), base * 1.25, grid);
    }
    let rain = fbm(vec2<f32>(input.world_position.x * 1.8 + input.world_position.z * 0.25, input.world_position.y * 0.13));
    let streaks = smoothstep(0.48, 0.86, rain) * smoothstep(14.0, 1.0, input.world_position.y);
    let grime = streaks * 0.10;
    if (material_uniform.material.x > 0.5 && material_uniform.material.x < 5.5) { base *= 1.0 - grime; }

    if (material_uniform.material.x > 3.5 && material_uniform.material.x < 4.5) {
        let cell = floor(input.uv * vec2<f32>(7.0, 18.0));
        let local = fract(input.uv * vec2<f32>(7.0, 18.0));
        let frame = step(0.08, local.x) * step(local.x, 0.92) * step(0.10, local.y) * step(local.y, 0.78);
        let lit = step(0.57, hash21(cell + floor(input.world_position.xz)));
        let warm = mix(vec3<f32>(0.55, 0.72, 1.0), vec3<f32>(1.0, 0.62, 0.25), hash21(cell + 9.1));
        base = mix(base * vec3<f32>(0.05, 0.08, 0.11), warm * 0.42, frame * lit * 0.30);
    }

    if (material_uniform.material.x > 2.5 && material_uniform.material.x < 3.5) {
        base = mix(base, vec3<f32>(0.015, 0.030, 0.045), 0.55);
        base += vec3<f32>(0.10, 0.045, 0.018) * step(0.82, hash21(floor(input.world_position.xy * 2.0)));
    }
    if (material_uniform.material.x > 4.5 && material_uniform.material.x < 5.5) {
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
    let backlit = smoothstep(0.18, 0.92, dot(-n, sun)) * pow(1.0 - max(dot(n, v), 0.0), 2.2);
    let silhouette_edge = smoothstep(0.10, 0.82, 1.0 - abs(dot(n, v)));
    let rim_shadow_gate = mix(0.55, 1.0, shadow);
    let rim = backlit * silhouette_edge * rim_shadow_gate;
    let direct = uniforms.sun_color.rgb * diffuse * direct_shadow * (1.48 - m.x * 0.34);
    var color = base * (sky_ambient + cool_bounce + direct) + reflection + uniforms.sun_color.rgb * (spec + rim * 1.05);

    let emissive = textureSample(emissive_texture, material_sampler, input.uv).rgb * material_uniform.material.w;
    if (m.w > 0.5) { color += base * m.z * 0.58; }
    color += emissive * (1.15 + m.z * 0.35);
    if (material_uniform.material.x < 0.5) {
        let lane_reflect = pow(max(dot(reflect(-sun, n), v), 0.0), 26.0);
        color += uniforms.sun_color.rgb * lane_reflect * 0.55 * direct_shadow;
        color *= 0.82 + 0.18 * hash21(input.world_position.xz * 8.0);
    }

    let height_fog = clamp(1.25 - input.world_position.y / 16.0, 0.16, 1.0);
    let depth_bias = smoothstep(14.0, 112.0, dist);
    let view_dir = normalize(input.world_position - uniforms.camera_position.xyz);
    let sun_view = pow(max(dot(view_dir, sun), 0.0), 7.0);
    let canyon_dust = smoothstep(2.0, 24.0, input.world_position.y) * smoothstep(120.0, 28.0, dist);
    let humid_noise = 0.86 + 0.20 * fbm(input.world_position.xz * 0.055 + uniforms.settings.xx * 0.006);
    let aerial = (1.0 - exp(-dist * uniforms.settings.z * 0.72 * height_fog * humid_noise)) * depth_bias;
    let horizon_fog = smoothstep(-0.08, 0.16, view_dir.y) * smoothstep(0.42, -0.06, view_dir.y);
    let fog = clamp(aerial + horizon_fog * 0.16 + canyon_dust * 0.035, 0.0, 0.82);
    let cool_distant_shadow = vec3<f32>(0.12, 0.14, 0.25);
    let warm_haze = mix(cool_distant_shadow, uniforms.horizon_color.rgb * 0.58 + uniforms.sun_color.rgb * (0.28 + sun_view * 0.72), 0.62);
    color = mix(color, warm_haze, fog);
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

    let ndc = vec4<f32>(input.uv * 2.0 - vec2<f32>(1.0), 1.0, 1.0);
    let world = uniforms.inv_view_proj * ndc;
    let view_dir = normalize(world.xyz / world.w - uniforms.camera_position.xyz);
    let sun = normalize(uniforms.sun_direction.xyz);
    let sun_clip = uniforms.view_proj * vec4<f32>(uniforms.camera_position.xyz + sun * 200.0, 1.0);
    let sun_ndc = sun_clip.xy / max(sun_clip.w, 0.0001);
    let sun_uv = sun_ndc * 0.5 + vec2<f32>(0.5);
    let sun_on_screen = step(0.0, sun_clip.w) * step(-0.08, sun_uv.x) * step(sun_uv.x, 1.08) * step(-0.08, sun_uv.y) * step(sun_uv.y, 1.08);
    let look_sun = smoothstep(0.950, 0.998, dot(view_dir, sun));
    let sun_scene_uv = vec2<f32>(sun_uv.x, 1.0 - sun_uv.y);

    var source_peak = 0.0;
    var source_dark = 0.0;
    for (var sx: i32 = -2; sx <= 2; sx = sx + 1) {
        for (var sy: i32 = -2; sy <= 2; sy = sy + 1) {
            let tap_uv = sun_scene_uv + vec2<f32>(f32(sx), f32(sy)) * texel * 5.0;
            let tap_luma = dot(textureSample(hdr_scene, hdr_sampler, tap_uv).rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            source_peak = max(source_peak, tap_luma);
            source_dark += (1.0 - smoothstep(0.18, 1.20, tap_luma)) / 25.0;
        }
    }
    let source_contrast = smoothstep(0.10, 0.68, source_dark) * smoothstep(1.25, 5.5, source_peak);
    let visible_source = sun_on_screen * smoothstep(1.15, 6.5, source_peak) * (0.45 + source_contrast * 0.85);

    let to_sun = sun_scene_uv - scene_uv;
    let ray_len = length(to_sun);
    let ray_dir = to_sun / max(ray_len, 0.0001);
    let tangent = vec2<f32>(-ray_dir.y, ray_dir.x);
    let dither = hash21(floor(input.uv * dims) + vec2<f32>(uniforms.settings.x * 23.17, uniforms.settings.x * 9.31));
    var shafts = vec3<f32>(0.0);
    var transmittance = 1.0;
    var edge_energy = 0.0;
    var weight_sum = 0.0;
    for (var i: i32 = 0; i < 48; i = i + 1) {
        let fi = f32(i);
        let t = (fi + 0.35 + dither * 0.55) / 48.0;
        let falloff = pow(1.0 - t, 1.45);
        let width = mix(0.65, 3.20, t) * texel * (1.0 + ray_len * 1.4);
        let base_uv = scene_uv + to_sun * t;
        let tap_a = textureSample(hdr_scene, hdr_sampler, base_uv).rgb;
        let tap_b = textureSample(hdr_scene, hdr_sampler, base_uv + tangent * width).rgb;
        let tap_c = textureSample(hdr_scene, hdr_sampler, base_uv - tangent * width).rgb;
        let luma_a = dot(tap_a, vec3<f32>(0.2126, 0.7152, 0.0722));
        let luma_b = dot(tap_b, vec3<f32>(0.2126, 0.7152, 0.0722));
        let luma_c = dot(tap_c, vec3<f32>(0.2126, 0.7152, 0.0722));
        let beam_luma = max(max(luma_a, luma_b), luma_c);
        let local_min = min(min(luma_a, luma_b), luma_c);
        let local_edge = smoothstep(0.20, 2.35, beam_luma - local_min);
        let blocker = 1.0 - smoothstep(0.12, 0.72, (luma_a + luma_b + luma_c) * 0.3333);
        transmittance *= mix(0.985, 0.900, blocker * (1.0 - t * 0.55));
        let bright = max(beam_luma - 0.92, 0.0);
        shafts += uniforms.sun_color.rgb * bright * transmittance * falloff * (0.012 + local_edge * 0.020);
        edge_energy += local_edge * falloff;
        weight_sum += falloff;
    }
    shafts /= max(weight_sum * 0.23, 0.0001);
    let radial = 1.0 - smoothstep(0.04, 0.86, distance(input.uv, sun_uv));
    let partial_occlusion = smoothstep(0.09, 0.70, edge_energy / max(weight_sum, 0.0001)) * smoothstep(0.98, 0.18, transmittance);
    color += shafts * visible_source * radial * (0.26 + partial_occlusion * 1.65);
    color += uniforms.sun_color.rgb * look_sun * visible_source * 0.16;

    let flare_axis = input.uv - 0.5;
    let sun_axis = sun_uv - 0.5;
    let ghost1 = 1.0 - smoothstep(0.00, 0.052, distance(flare_axis, -sun_axis * 0.42));
    let ghost2 = 1.0 - smoothstep(0.00, 0.032, distance(flare_axis, -sun_axis * 0.82));
    let anamorphic = exp(-abs(input.uv.y - sun_uv.y) * 94.0) * smoothstep(0.70, 0.0, abs(input.uv.x - sun_uv.x));
    color += (uniforms.sun_color.rgb * ghost1 * 0.026 + vec3<f32>(0.45,0.58,1.0) * ghost2 * 0.014 + uniforms.sun_color.rgb * anamorphic * 0.014) * visible_source;

    var bloom = vec3<f32>(0.0);
    for (var x: i32 = -2; x <= 2; x = x + 1) {
        for (var y: i32 = -2; y <= 2; y = y + 1) {
            let s = textureSample(hdr_scene, hdr_sampler, scene_uv + vec2<f32>(f32(x), f32(y)) * texel * 2.0).rgb;
            bloom += max(s - vec3<f32>(1.75), vec3<f32>(0.0)) / 25.0;
        }
    }
    color += bloom * 0.24 * uniforms.settings.w;
    color *= uniforms.settings.y * 1.03;
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = mix(color * vec3<f32>(0.88, 0.94, 1.06), color * vec3<f32>(1.08, 0.98, 0.88), smoothstep(0.35, 1.8, luma));
    let d = distance(input.uv, vec2<f32>(0.5));
    color *= 1.0 - smoothstep(0.52, 0.92, d) * 0.045;
    color = aces_tonemap(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(color, 1.0);
}
