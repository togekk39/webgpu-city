struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    time_effects: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var world = input.position;
    if (input.color.a > 0.9 && input.color.a < 1.1) {
        let direction = select(-1.0, 1.0, input.position.x < 0.0);
        let lane_speed = 2.8 + abs(input.position.x) * 0.08;
        world.z = fract((input.position.z + 50.0 + uniforms.time_effects.x * lane_speed * direction) / 100.0) * 100.0 - 50.0;
    }
    if (input.color.a > 1.9 && input.color.a < 2.1) {
        world.x = input.position.x + sin(uniforms.time_effects.x * 0.9 + input.position.z) * 0.18;
        world.y = input.position.y + abs(sin(uniforms.time_effects.x * 2.4 + input.position.x)) * 0.035;
    }
    out.clip_position = uniforms.view_proj * vec4<f32>(world, 1.0);
    out.world_position = world;
    out.color = input.color;
    return out;
}

fn filmic_tonemap(c: vec3<f32>) -> vec3<f32> {
    let x = max(vec3<f32>(0.0), c - vec3<f32>(0.004));
    return (x * (6.2 * x + vec3<f32>(0.5))) / (x * (6.2 * x + vec3<f32>(1.7)) + vec3<f32>(0.06));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_dx = dpdx(input.world_position);
    let world_dy = dpdy(input.world_position);
    let normal = normalize(cross(world_dx, world_dy));
    let sun = normalize(uniforms.light_dir.xyz);
    let ndl = clamp(dot(normal, sun), 0.0, 1.0);
    let sky = vec3<f32>(0.055, 0.075, 0.12) * (0.55 + normal.y * 0.35);
    let sunset = vec3<f32>(1.0, 0.43, 0.11) * pow(clamp(dot(normal, sun) * 0.5 + 0.5, 0.0, 1.0), 3.0);
    let contact = 1.0 - (1.0 - smoothstep(0.0, 1.2, input.world_position.y)) * 0.36;
    let wet_road = (1.0 - smoothstep(0.0, 0.18, input.world_position.y)) * (0.35 + 0.65 * pow(abs(normal.y), 8.0));
    let is_emissive = select(0.0, 1.0, max(max(input.color.r, input.color.g), input.color.b) > 0.78);
    let flicker = 0.88 + 0.12 * sin(uniforms.time_effects.x * 2.7 + input.world_position.x * 1.7 + input.world_position.z * 0.43);
    var lit = input.color.rgb * (sky + sunset * 0.95 + vec3<f32>(ndl * 0.55)) * contact;
    lit += input.color.rgb * is_emissive * (1.8 + uniforms.time_effects.z * 0.8) * flicker;
    lit += vec3<f32>(1.0, 0.52, 0.18) * wet_road * 0.18;
    lit += vec3<f32>(0.9, 0.33, 0.08) * pow(max(dot(normal, sun), 0.0), 12.0) * wet_road;

    let dist = length(input.world_position.xz - vec2<f32>(-18.0, 31.0));
    let height_haze = clamp((input.world_position.y + 2.0) / 34.0, 0.0, 1.0);
    let fog = clamp((dist - 18.0) / 82.0, 0.0, 1.0) * uniforms.time_effects.y;
    let fog_color = mix(vec3<f32>(0.035, 0.048, 0.075), vec3<f32>(0.95, 0.38, 0.12), height_haze * 0.55);
    lit = mix(lit, fog_color, fog * 0.58);
    let graded = filmic_tonemap(lit * 1.35);
    let cinematic = pow(graded, vec3<f32>(0.92, 0.96, 1.05)) * vec3<f32>(1.06, 0.94, 0.86);
    return vec4<f32>(mix(graded, cinematic, uniforms.time_effects.w), 1.0);
}
