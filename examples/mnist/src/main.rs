use std::fs::File; 
use std::io::Cursor; 
use std::io::Read;
use byteorder::*; 

mod display; 
use display::to_ppm_file; 

fn main() -> Result<(), std::io::Error> {
    let path = "data/";
    let data_set = load_mnist_from_path(path)?; 

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

    pub fn new_from_bytes(bytes : &[u8]) -> Result<Self, std::io::Error> { 

        let mut reader = Cursor::new(&bytes); 
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
            _ => panic!("Magic number not recognized as number for MNIST dataset")
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
pub struct MnistDataset { 
    train_set : Vec<MnistImage>, 
    test_set : Vec<MnistImage>
}


pub fn load_mnist_from_bytes(train_data : &[u8], train_labels : &[u8], test_data : &[u8], test_labels : &[u8]) -> Result<MnistDataset, std::io::Error>{ 

    let data_processing  = |data : &[u8], labels : &[u8]| -> std::result::Result<Vec<MnistImage>, std::io::Error> { 
        println!("readin");
        let image_data = MnistFileData::new_from_bytes(data)?; 
        let label_data = MnistFileData::new_from_bytes(labels)?; 

        assert_eq!(image_data.sizes[0],label_data.sizes[0], "Image and label data sizes are not equal");

        let image_length = (image_data.sizes[1]*image_data.sizes[2]) as usize;

        let out : Vec<MnistImage> = image_data.data
        .windows(image_length)
        .zip(label_data.data)
        .map(|(mnist_img, mnist_label)| 
        MnistImage{
            data :  mnist_img.to_vec().into_iter().map(|x| x as f64/255.0).collect(), 
            label: mnist_label
        }).collect();

        Ok(out)
    };

    let train_set = data_processing(train_data, train_labels)?;
    let test_set = data_processing(test_data, test_labels)?;

    Ok(MnistDataset{train_set, test_set})
}


fn load_mnist_from_path(path : &str) -> Result<MnistDataset, std::io::Error> {
    let mut train_images_file = File::open(format!("{}/train-images.idx3-ubyte", path))?; 
    let mut train_labels_file = File::open(format!("{}/train-labels.idx1-ubyte", path))?; 
    let mut test_images_file = File::open(format!("{}/t10k-images.idx3-ubyte", path))?; 
    let mut test_labels_file = File::open(format!("{}/t10k-labels.idx1-ubyte", path))?; 

    let extract_file = |f : &mut File| -> std::result::Result<std::vec::Vec<u8>, std::io::Error> {
        let mut output : Vec<u8> = Vec::new(); 
        f.read_to_end(&mut output)?;
        Ok(output)
    };

    let train_data = extract_file(&mut train_images_file)?;
    let train_labels = extract_file(&mut train_labels_file)?;
    let test_data = extract_file(&mut test_images_file)?; 
    let test_labels = extract_file(&mut test_labels_file)?;

    load_mnist_from_bytes(&train_data, &train_labels, &test_data, &test_labels)
}