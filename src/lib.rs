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

pub const MAX_STEPS: usize = 500;
pub const MAX_DIST: f64 = 100.;
pub const SURF_DIST: f64 = 0.0001;
pub const TAU: f64 = PI * 2.0;
pub const STEP_SCALE: f64 = 1.0;

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
                            let p = ro + d * rd;
                            color += simple_shading(p, rd);
                        }
                    }
                    color *= sample_scale;
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

pub fn ray_march(ro: Point, rd: Vector) -> f64 {
    let mut d = 0.0;

    for _ in 0..MAX_STEPS {
        let p = ro + rd * d;
        let ds = eval(p);
        d += ds * STEP_SCALE;
        if d >= MAX_DIST || ds < SURF_DIST {
            break;
        }
    }
    return d;
}

fn simple_shading(p: Point, rd: Vector) -> Color {
    let n = gradient(p);
    let mut color = vector![0.2,0.8,1.];

    // lighting
    let light1 = n.dot(&vector![1., -1., -1.].normalize())*0.5+0.5;
    let light2 = n.dot(&vector![-1., -1., -1.].normalize())*0.5+0.5;
    let illumination = 0.5 * light1 + 0.5 * light2;

    // fake fresnel
    let n_dot_v = n.dot(&rd) + 1.;
    let fresnel = n_dot_v * n_dot_v * 0.45;
    
    // Specular highlights
    let r = reflect(vector![1., 0., 0.].normalize(), n);
    let specular = vec3(1.0) * r.dot(&rd).max(0.0).powf(10.0) * 0.08;

    color *= illumination;
    color += vec3(fresnel);
    color += specular;

    return color;

}

// Environment
pub fn sky(uv: (f64, f64)) -> Color {
    // Background 
    let mut c = vector![0.1,0.7,1.];

    c += vec3(lerp(0.2, 0.4, 1.0 - uv.1));
    c += vec3(lerp(0.2, 0.4, uv.0));

    return c
}

////////////////////////////////////////////////////////////////
// Main Scene SDF
////////////////////////////////////////////////////////////////

pub fn eval(p: Point) -> f64 {
    let s1 = _sphere(p, point![0.0, 0.0, 0.0], 1.);
    let s2 = _sphere(p, point![1., -0.6, -1.], 0.9);
    // return _boolean_union(s1, s2);
    return _smooth_subtraction(s1, s2, 0.05);
    // return _smooth_intersection(s1, s2, 0.5);
}

////////////////////////////////////////////////////////////////
// Signed Distance Fields
////////////////////////////////////////////////////////////////
// Main reference: https://iquilezles.org/articles/distfunctions/

pub fn _sphere(p: Point, c: Point, r: f64) -> f64 {
    return distance(&p, &c) - r;
}

fn _rounded_box(p: Point, s: Vector, r: f64) -> f64 {
    // Modified to account for the radius without changing the size of the box
    let pf = vector![p.x.abs(), p.y.abs(), p.z.abs()] - (s-vec3(r));
    return vector![pf.x.max(0.0),pf.y.max(0.0),pf.z.max(0.0)].norm() + pf.x.max(pf.y.max(pf.z)).min(0.0) - r;
}

fn _rounded_cylinder(p: Point, r1: f64, r2: f64, h: f64) -> f64 {
    let d = vector![vector![p.x, p.z].norm()-2.0*r1+r2, p.y.abs() - h];
    return d.x.max(d.y).min(0.0) + vector![d.x.max(0.0), d.y.max(0.0)].norm() - r2
}

fn _torus( p: Point, r1: f64, r2: f64 ) -> f64 {
  let q = vector![vector![p.x, p.z].norm()-r1, p.y];
  return q.norm()-r2;
}

// Gradient of a Signed Distance Field
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

////////////////////////////////////////////////////////////////
// SDF Operations
////////////////////////////////////////////////////////////////

fn _boolean_union(d1: f64, d2: f64) -> f64 {
    return d2.min(d1)
}

fn _boolean_subtraction(d1: f64, d2: f64) -> f64 {
    return (-d2).max(d1)
}

fn _boolean_intersection(d1: f64, d2: f64) -> f64 {
    return d1.max(d2)
}

fn _smooth_union(d1: f64, d2: f64, k: f64) -> f64 {
    let h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return lerp( d2, d1, h ) - k*h*(1.0-h);
}

fn _smooth_subtraction(d1: f64, d2: f64, k: f64) -> f64 {
    let h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return lerp( d1, -d2, h ) + k*h*(1.0-h);
}

fn _smooth_intersection(d1: f64, d2: f64, k: f64) -> f64 {
    let h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return lerp( d2, d1, h ) + k*h*(1.0-h);
}

////////////////////////////////////////////////////////////////
// Vectors & Transforms
////////////////////////////////////////////////////////////////

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

////////// Rotations //////////

fn _rot_x(p: Point, a: f64) -> Point {
    let s = a.sin();
    let c = a.cos();

    return point![p.x, p.y * c + p.z * s, -s * p.y + c * p.z]
}

fn _rot_y(p: Point, a: f64) -> Point {
    let s = a.sin();
    let c = a.cos();
    
    return point![c * p.x + s * p.z, p.y, -s * p.x + c * p.z]
}

fn _rot_z(p: Point, a: f64) -> Point {
    let s = a.sin();
    let c = a.cos();

    return point![c * p.x - s * p.y, s * p.x + c * p.y, p.z]
}

////////// Other Math //////////
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

fn clamp(x: f64, a: f64, b: f64) -> f64 {
    x.max(a).min(b)
}

////////////////////////////////////////////////////////////////
// Image Writing
////////////////////////////////////////////////////////////////

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
