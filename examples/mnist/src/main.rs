use std::borrow::BorrowMut;
use std::fs::File; 
use std::io::Cursor; 
use std::io::Read;
use byteorder::*; 

mod display; 
use display::to_ppm_file; 

fn main() -> Result<(), std::io::Error> {
    let path = "data/";
    let data_set = load_mnist_dataset(path)?; 

    let out_path = "output/first.ppm";
    to_ppm_file(out_path, &data_set.train_set[0])?;

    Ok(())
}

#[derive(Debug, Clone)]
struct MnistFileData {
    sizes : Vec<i32>,
    data : Vec<u8>
}


impl MnistFileData { 
    pub fn new(file : &mut File) -> Result<Self, std::io::Error> { 

        let mut contents : Vec<u8> = Vec::new(); 
        file.read_to_end(&mut contents)?; 

        let mut reader = Cursor::new(&contents); 
        let magic_number = reader.read_i32::<BigEndian>()?;

        let mut sizes : Vec<i32> = Vec::new();
        let mut data : Vec<u8> = Vec::new(); 

        match magic_number { 

            // This magic number corresponds to label data, where there is one 32-byte that corresponds to the # of samples 
            // in the file
            2049 => {
                sizes.push(reader.read_i32::<BigEndian>()?);
            }
            // This magic number corresponds to a file that has image files in it
            2051 => { 
                sizes.push(reader.read_i32::<BigEndian>()?);
                sizes.push(reader.read_i32::<BigEndian>()?);
                sizes.push(reader.read_i32::<BigEndian>()?);
            }
            _ => panic!("Problem reading file after magic number")
        }
        
        reader.read_to_end(&mut data)?;

        Ok(MnistFileData{sizes, data})
    }
}


/// Struct that represents a single MNIST image. That is 
#[derive(Debug, Clone)]
pub struct MnistImage { 
    data : Vec<f64>, 
    label : u8
}


/// Structure that represents the data of the mnist dataset
#[derive(Debug, Clone)]
struct MnistDataset { 
    train_set : Vec<MnistImage>, 
    test_set : Vec<MnistImage>
}



fn load_mnist_dataset(path : &str) -> Result<MnistDataset, std::io::Error>{ 
    let file_prefixes = ["train", "t10k"];

    let mut train_set : Vec<MnistImage> = Vec::new();
    let mut test_set : Vec<MnistImage> = Vec::new();


    for prefix in file_prefixes {
        let image_filename = format!("{}/{}-images.idx3-ubyte", path, prefix); 
        let label_filename = format!("{}/{}-labels.idx1-ubyte", path, prefix); 

        let image_data = MnistFileData::new(&mut File::open(&image_filename)?)?; 
        let label_data = MnistFileData::new(&mut File::open(&label_filename)?)?; 

        assert_eq!(image_data.sizes[0],label_data.sizes[0], "Image and label data sizes are not equal");

        // let number_images = image_data.sizes[0] as usize; 
        let image_length = (image_data.sizes[1]*image_data.sizes[2]) as usize;

        let out : Vec<MnistImage> = image_data.data
        .windows(image_length)
        .zip(label_data.data)
        .map(|(mnist_img, mnist_label)| 
        MnistImage{
            data :  mnist_img.to_vec().into_iter().map(|x| x as f64/255.0).collect(), 
            label: mnist_label
        }).collect();

        match prefix {
            "train" => {train_set = out}
            "t10k" => {test_set = out}
            _ => {panic!("Unexpected match while parsing mnist files")}
        }
    }
    Ok(MnistDataset{
        train_set, 
        test_set
    })
}