use mnist_load::load_mnist_from_path;

mod display; 
use display::to_ppm_file; 

fn main() -> Result<(), std::io::Error> {
    let path = "data/";
    let data_set = load_mnist_from_path(path)?; 

    let out_path = "output/first.ppm";
    to_ppm_file(out_path, &data_set.train_set[0])?;

    Ok(())
}

