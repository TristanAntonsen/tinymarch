use tinymarch::render;
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let x_res = 1080;
    let y_res = 1080;
    let sample_count = 20; // anti-aliasing

    render(x_res, y_res, sample_count);

    let elapsed = now.elapsed().as_secs_f64();
    let s = elapsed % 60.;
    let min = (elapsed / 60.).floor() as u8;
    println!("\n{} min {:.2?} seconds", min, s);
}
