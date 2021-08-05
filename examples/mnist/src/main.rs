use neural_net::BackPropOutput;
use neural_net::F64Vector;
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
        ActivationFunction::Sigmoid)
}

fn label_to_vector(label : u8) -> F64Vector { 

    match label  { 
        0 => {F64Vector::from_vec(vec![1.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        1 => {F64Vector::from_vec(vec![0.0, 1.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        2 => {F64Vector::from_vec(vec![0.0, 0.0 , 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        3 => {F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        4 => {F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        5 => {F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])}
        6 => {F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
        7 => {F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])}
        8 => {F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])}
        9 => {F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])}
        _ => {panic!("Unexpected label value")}
        
    }
}


fn learning_epoch(nn : &mut NeuralNetwork, mnist_data : &MnistDataset) { 

    let batch_size = 100; 
    let mut backprop_accum: Option<BackPropOutput> = None;

    let mut batch_idx : u64 = 0;

    let mut err_accum = F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // let normalizing_factor = (1/&mnist_data.train_set.len()) as f64;
    let normalizing_factor = 1.0/20.0;

    for mnist_img in &mnist_data.train_set[0..10000] { 

        let out = nn.propagate_vec(mnist_img.data.clone());
        let expected = label_to_vector(mnist_img.label); 

        let diff = &expected - &out;

        err_accum = err_accum + ((&diff.component_mul(&diff))* normalizing_factor);

        let back_prop_out = nn.backprop(diff); 

        backprop_accum = match backprop_accum {
            None => Some(back_prop_out),
            Some(accum) => Some(NeuralNetwork::combine_backprop_outputs(
                accum,
                &back_prop_out,
            )),
        };

        if batch_idx % batch_size == 0 { 

            println!("Updating network..  on sample {}", batch_idx);
            let update = backprop_accum.take().unwrap();
            nn.update_weights(&update);
            err_accum = F64Vector::from_vec(vec![0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        batch_idx += 1; 

    }

    println!{"Error for epoch {}", err_accum}

}