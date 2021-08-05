use std::fs::File; 
use std::io::Cursor; 
use std::io::Read;
use byteorder::*; 

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


/// Struct that represents a single MNIST image. That is a flat vector of each pixel value as a float from 0 to 1. 
/// The data label corresponding to the image is also provided where it can be any digits 0-9
#[derive(Debug, Clone)]
pub struct MnistImage { 
    pub data : Vec<f64>, 
    pub label : u8
}


/// Structure that represents the data of the MNIST Dataset. 
#[derive(Debug, Clone)]
pub struct MnistDataset { 
    /// 60000 MNIST images that are meant to be used to train machine learning algorithms
    pub train_set : Vec<MnistImage>, 
    /// 10000 MNIST images that are meant ot be used as a test set for machine learning algorithms
    pub test_set : Vec<MnistImage>
}

/// Loads the MNIST dataset from a set of byte arrays that are passed in. Each byte array must correspond to the bytes of reading a 
/// un-gzipped MNIST file that can be found in http://yann.lecun.com/exdb/mnist/
/// 
/// ### params 
/// * train_data: un-gzipped bytes of "train-images-idx3-ubyte.gz"
/// * train_labelse: un-gzipped bytes of "train-labels.idx1-ubyte"
/// * test_data: un-gzipped bytes of "t10k-images.idx3-ubyte"
/// * test_labels: un-gzipped bytes of "t10k-labels.idx1-ubyte"
///  
/// ## Panics
/// 
/// Panics any of the byte arrays don't follow the file format that is expected out of bytes that conform to the format as 
/// specified in the MNIST website
/// 
/// Panics if the number of training labels don't correspond to the same number of training images 
/// 
/// Panics if the number of test labels don't corresopnd to the same number of training images
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

/// Function that loads the MNIST dataset, returning the complete MNIST dataset as easy to use structures.
/// 
/// ## param
/// * path: A &str that corresponds to the path to the containing folder for the following files
/// 
/// 1. "train-images.idx3-ubyte"
/// 2. "train-labels.idx1-ubyte"
/// 3. "t10k-images.idx3-ubyte"
/// 4. "t10k-labels.idx1-ubyte"
/// 
/// ## Returns
/// A proper MNIST struct or an IO error
/// 
/// ## Panics
/// If the passed in files dont follow the MNIST data format as specified at the bottom of the page of http://yann.lecun.com/exdb/mnist/
/// 
/// Panics if the number of training labels don't correspond to the same number of training images 
/// 
/// Panics if the number of test labels don't corresopnd to the same number of training images
pub fn load_mnist_from_path(path : &str) -> Result<MnistDataset, std::io::Error> {
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