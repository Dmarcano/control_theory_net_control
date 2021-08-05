use mnist_load::{load_mnist_from_path, MnistDataset};
use neural_net::{NeuralNetwork, LayerTopology, activation_funcs::ActivationFunction}; 

mod display;
use display::to_ppm_file;

fn main() -> Result<(), std::io::Error> {
    let path = "data/";
    let mnist_data = load_mnist_from_path(path)?;

    let mut nn = mnist_nn();

    learning_epoch(&mut nn, &mnist_data);
    Ok(())
}


fn mnist_nn() -> NeuralNetwork { 
    let layer_one = LayerTopology{num_neurons : 28*28};
    let layer_two = LayerTopology{num_neurons: 300};
    let layer_three = LayerTopology{num_neurons: 10}; 

    NeuralNetwork::random(
        &[layer_one, layer_two, layer_three],
        Some(0.1), None,
        ActivationFunction::LeakyRelu)
}


fn learning_epoch(nn : &mut NeuralNetwork, mnist_data : &MnistDataset) { 

    unimplemented!()

}