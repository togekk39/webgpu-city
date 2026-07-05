struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.world_position = input.position;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_dx = dpdx(input.world_position);
    let world_dy = dpdy(input.world_position);
    let normal = normalize(cross(world_dx, world_dy));
    let light = normalize(uniforms.light_dir.xyz);
    let diffuse = clamp(dot(normal, light), 0.0, 1.0);
    let rim = pow(1.0 - clamp(abs(normal.y), 0.0, 1.0), 2.0) * 0.12;
    let glow = smoothstep(2.2, 7.8, input.world_position.y) * 0.22;
    let street = 1.0 - smoothstep(0.0, 0.45, input.world_position.y);
    let distance_fog = clamp(length(input.world_position.xz) / 28.0, 0.0, 1.0);
    let lit = input.color * (0.48 + diffuse * 0.52) + vec3<f32>(glow + rim, glow * 0.78, glow * 0.35) + vec3<f32>(street * 0.035, street * 0.03, street * 0.02);
    let sky_tint = vec3<f32>(0.04, 0.06, 0.10);
    return vec4<f32>(mix(lit, sky_tint, distance_fog * 0.32), 1.0);
}
