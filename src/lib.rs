use std::f64::consts::PI;

use image::{Rgb, RgbImage};
use nalgebra::{distance, point, vector, Point3, Vector3};
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

// ======================================================================
// ======================= Data types & Constants =======================
// ======================================================================

pub type Point = Point3<f64>;
pub type Vector = Vector3<f64>;
pub type Color = Vector3<f64>;

pub const MAX_STEPS: usize = 1000;
pub const MAX_DIST: f64 = 1000.;
pub const SURF_DIST: f64 = 0.001;
pub const TAU: f64 = PI * 2.0;

pub const BLACK: Color = vector![0.0, 0.0, 0.0];
pub const WHITE: Color = vector![1.0, 1.0, 1.0];
pub const RED: Color = vector![1.0, 0.0, 0.0];
pub const GREEN: Color = vector![0.0, 1.0, 0.0];
pub const BLUE: Color = vector![0.0, 0.0, 1.0];

// ======================================================================
// ==================== Main Loop & Render Functions ====================
// ======================================================================

pub fn render(res_x: usize, res_y: usize, samples: usize) {
    let ro = point![0., 0., -10.];
    let rt = point![0., 0., 0.];

    // light & environment
    let lights = vec![
        // (origin, power)
        (point![1.0, 1.0, -1.0], 16.),
        (point![-1.0, 1.0, -1.0], 2.0),
        (point![1.0, 1.0, -1.0], 2.),
        (point![-1.0, -1.0, -1.0], 2.0),
    ];

    // sampling
    let sample_scale = 1. / (samples as f64);

    let pixels = (0..res_x)
        .into_par_iter()
        .map(|x| {
            (0..res_y)
                .into_par_iter()
                .rev()
                .map(|y| {
                    let mut color = vec3(0.0);
                    let mut sampler = rand::thread_rng();
                    for _ in 0..samples {
                        // screen space coordinates
                        let u = (x as f64 + sampler.gen::<f64>()) / res_x as f64;
                        let v = 1.0 - (y as f64 + sampler.gen::<f64>()) / res_y as f64; // flip y

                        // ray direction
                        let rd = ray_direction((u - 0.5, v - 0.5), ro, rt, (res_x, res_y));

                        // ray marching
                        let d = ray_march(ro, rd);

                        // shading
                        if d >= MAX_DIST {
                            color += sky((u, v));
                        } else {
                            // intersection point & normal
                            let p = ro + d * rd;
                            // color += shading(ro, rd, p, &lights)
                            color += simple_shading(p);
                        }
                    }
                    color *= sample_scale;
                    // gamma_correct(color); // only if PBR shading
                    color
                })
                .collect::<Vec<Color>>()
        })
        .collect::<Vec<Vec<Color>>>();

    save_png(pixels, "output.png")
}

// Ray direction
fn ray_direction(uv: (f64, f64), ro: Point, rt: Point, res: (usize, usize)) -> Vector {

    // screen orientation
    let vup = vector![0., 1.0, 0.0];
    let aspect_ratio = (res.0 as f64) / (res.1 as f64);

    let vw = (ro - rt).normalize();
    let vu = (vup.cross(&vw)).normalize();
    let vv = vw.cross(&vu);
    let theta = 30. * 2. * PI / 180.; // half FOV
    let viewport_height = 2. * (theta.tan());
    let viewport_width = aspect_ratio * viewport_height;
    let horizontal = -viewport_width * vu;
    let vertical = viewport_height * vv;
    let focus_dist = (ro - rt).norm();
    let center = ro - vw * focus_dist;

    let rd = center + uv.0 * horizontal + uv.1 * vertical - ro;

    return rd.normalize();
}

fn simple_shading(p: Point) -> Color {
    let n = gradient(p);

    // let light1 = dot(N, normalize(vec3(1,-1,-1)))*.5+.5;
    let light1 = n.dot(&vector![1., -1., -1.].normalize())*0.5+0.5;
    let light2 = n.dot(&vector![-1., -1., -1.].normalize())*0.5+0.5;

    let color = 0.5 * vec3(light1) + 0.5 * vec3(light2);

    return color;

}

// shading
fn shading(_ro: Point, rd: Vector, p: Point, lights: &Vec<(Point, f64)>) -> Color {
    // material parameters
    let albedo = vector![1.0, 0.0, 0.0];
    let roughness = 0.5;
    let metallic = 0.0;
    let mut f0 = vec3(0.04);
    f0 = mix_vectors(f0, albedo, metallic);

    let n = gradient(p);

    let mut lo = vec3(0.0);
    for light in lights {
        // reflectance equation
        // radiance
        let v = -rd;
        let l = (light.0 - p).normalize();
        let h = (v + l).normalize();
        let dist = (light.0 - p).norm();
        let attenuation = 1.0 / (dist * dist);
        let radiance = vec3(1.0) * attenuation * light.1;

        // brdf (cook-torrance)
        let ndf = distribution_ggx(n, h, roughness);
        let g = geometry_smith(n, v, l, roughness);
        let f = fresnel_schlick(h.dot(&v).max(0.0), f0);

        let ks = f;
        let kd = (vec3(1.0) - ks) * (1.0 - metallic);

        let numerator = ndf * g * f;

        let n_dot_l = n.dot(&l).max(0.0);
        let denomenator = 4.0 * n.dot(&v).max(0.0) * n_dot_l + 0.0001;

        let specular = numerator / denomenator;

        lo += multiply_vectors(
            multiply_vectors(kd, albedo) / PI + specular,
            // multiply_vectors(kd, diffuse(ro, rd)) / PI + specular,
            radiance * n_dot_l,
        );
    }

    let ambient = albedo * 0.03;
    return ambient + lo;
}

// Fresnel Equation - Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f64, f0: Vector) -> Vector {
    f0 + (vec3(1.0) - f0) * clamp(1.0 - cos_theta, 0.0, 1.0).powf(5.0)
}

// Normal Distribution Function
fn distribution_ggx(n: Vector, h: Vector, roughness: f64) -> f64 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = n.dot(&h).max(0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let num = a2;
    let mut denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;

    return num / denom;
}

// Geometry Function
fn geometry_smith(n: Vector, v: Vector, l: Vector, roughness: f64) -> f64 {
    let n_dot_v = n.dot(&v).max(0.0);
    let n_dot_l = n.dot(&l).max(0.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);

    ggx1 * ggx2
}
// Geometry Schlick GGX
fn geometry_schlick_ggx(n_dot_v: f64, roughness: f64) -> f64 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    let num = n_dot_v;
    let denom = n_dot_v * (1.0 - k) + k;

    num / denom
}

pub fn eval(p: Point) -> f64 {
    let s1 = sphere(p, point![0.0, 0.0, 0.0], 1.);
    let s2 = sphere(p, point![1., -0.6, -1.], 0.9);
    return boolean_subtraction(s1, s2);
}

pub fn sphere(p: Point, c: Point, r: f64) -> f64 {
    return distance(&p, &c) - r;
}

fn boolean_subtraction(d1: f64, d2: f64) -> f64 {
    return (-d2).max(d1);
}

pub fn gradient(p: Point) -> Vector {
    let epsilon = 0.0001;
    let dx = Vector3::new(epsilon, 0., 0.);
    let dy = Vector3::new(0., epsilon, 0.);
    let dz = Vector3::new(0., 0., epsilon);

    // Gradient: dSDF/dx, dy, dz
    let ddx = eval(p + dx) - eval(p - dx);
    let ddy = eval(p + dy) - eval(p - dy);
    let ddz = eval(p + dz) - eval(p - dz);

    vector![ddx, ddy, ddz].normalize()
}

pub fn ray_march(ro: Point, rd: Vector) -> f64 {
    let mut d = 0.0;

    for _ in 0..MAX_STEPS {
        let p = ro + rd * d;
        let ds = eval(p);
        d += ds;
        if d >= MAX_DIST || ds < SURF_DIST {
            break;
        }
    }
    return d;
}

// Environment
pub fn sky(uv: (f64, f64)) -> Color {
    // Background 
    let v = vector![uv.0, uv.1].norm() * 0.75;

    return vec3(lerp(0.1, 0.2, uv.1))
    // let fragColor = vec4f(mix(0.1, 0.2, smoothstep(0.0, 1.0, uv.y)));
}

pub fn save_png(pixels: Vec<Vec<Color>>, path: &str) {
    let width = pixels.len() as u32;
    let height = pixels[0].len() as u32;

    let mut img = RgbImage::new(width, height);
    for x in 0..width {
        for y in 0..height {
            let color = pixels[x as usize][y as usize];
            let r = (color[0] * 255.0).round() as u8;
            let g = (color[1] * 255.0).round() as u8;
            let b = (color[2] * 255.0).round() as u8;

            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    println!("{} exported.", path);

    img.save(path).expect("Could not save png");
}

////////// Colors //////////

fn gamma_correct(mut color: Color) -> Color {
    color = divide_vectors(color, color + vec3(1.0));
    powf_vector(color, 1.0 / 2.2)
}

////////// Vectors //////////

// create a Vector3 with constant values
pub fn vec3(a: f64) -> Vector {
    vector![a, a, a]
}

//reflect an input vector about another (the sdf surface normal)
pub fn reflect(v: Vector, normal: Vector) -> Vector {
    return v - normal * 2.0 * v.dot(&normal);
}

pub fn mix_vectors(v1: Vector, v2: Vector, t: f64) -> Vector {
    vector![
        lerp(v1.x, v2.x, t),
        lerp(v1.y, v2.y, t),
        lerp(v1.z, v2.z, t)
    ]
}

pub fn vector_max(v1: Vector, v2: Vector) -> Vector {
    vector![v1.x.max(v2.x), v1.y.max(v2.y), v1.z.max(v2.z)]
}

pub fn multiply_vectors(v1: Vector, v2: Vector) -> Vector {
    vector![v1.x * v2.x, v1.y * v2.y, v1.z * v2.z]
}

pub fn divide_vectors(v1: Vector, v2: Vector) -> Vector {
    vector![v1.x * v2.x, v1.y * v2.y, v1.z * v2.z]
}

pub fn powf_vector(v: Vector, p: f64) -> Vector {
    vector![v.x.powf(p), v.y.powf(p), v.z.powf(p)]
}

////////// Math //////////
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

fn clamp(x: f64, a: f64, b: f64) -> f64 {
    x.max(a).min(b)
}
