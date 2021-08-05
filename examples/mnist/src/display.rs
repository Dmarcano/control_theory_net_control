use mnist_load::MnistImage;
use std::fs::File;
use std::io::Write;

pub fn to_ppm_file(path: &str, image: &MnistImage) -> Result<(), std::io::Error> {
    let mut f = File::create(path)?;

    f.write("P3\n".as_bytes())?;
    f.write("28 28\n".as_bytes())?;
    f.write("255\n".as_bytes())?;

    let img_width = 28;
    let img_height = 28;

    for i in 0..img_height {
        for j in 0..img_width {
            let idx = (i * img_width) + j;
            let regularized_val = (255.0 * image.data[idx]) as u8;

            f.write(
                format!(
                    "{} {} {}\n",
                    regularized_val, regularized_val, regularized_val
                )
                .as_bytes(),
            )?;
        }
    }
    Ok(())
}
