use tinymarch::render;
use std::time::Instant;

fn main() {
    let now = Instant::now();
    render(720, 720, 16);
    let elapsed = now.elapsed().as_secs_f64();
    let s = elapsed % 60.;
    let min = (elapsed / 60.).floor() as u8;
    println!("\n{} min {:.2?} seconds", min, s);
}
