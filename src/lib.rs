use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage, Pixel};
use nalgebra::{vector, Point3, Vector3, point};

// ======================================================================
// ======================= Data types & Constants =======================
// ======================================================================

pub type Point = Point3<f64>;
pub type Vector = Vector3<f64>;
pub type Color = Vector3<f64>;

pub const BLACK: Color = vector![0.0, 0.0, 0.0];
pub const WHITE: Color = vector![1.0, 1.0, 1.0];
pub const RED: Color = vector![1.0, 0.0, 0.0];
pub const GREEN: Color = vector![0.0, 1.0, 0.0];
pub const BLUE: Color = vector![0.0, 0.0, 1.0];

pub fn render(res_x: usize, res_y: usize) {

    let ro = point![0., 0., 0.];
    let viewport_width = 1.0;
    let viewport_height = res_y as f64 / res_x as f64;    

    // screen orientation
    let horizontal = vector![viewport_width, 0., 0.];
    let vertical = vector![0., viewport_height, 0.];
    let focal_length = 1.0;
    let lower_left_corner = ro - 0.5 * horizontal - 0.5 * vertical - vector![0.0, 0.0, -focal_length];

    let pixels = (0..res_x)
        .into_iter()
        .map(|x| {
            (0..res_y)
                .into_iter()
                .map(|y| {

                    let u = x as f64 / res_x as f64;
                    let v = y as f64 / res_y as f64;
                    
                    let rd = (lower_left_corner + u*horizontal + v*vertical - ro).normalize();
                    let t = 0.5*(rd.y + 1.0);
                    (1.0-t) * vec3(1.0) + t * vector!(0.5, 0.7, 1.0)
                })
                .collect::<Vec<Color>>()
        })
        .collect::<Vec<Vec<Color>>>();

    save_png(pixels, "output.png")
}

pub fn vec3(a: f64) -> Vector {
    vector![a, a, a]
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
