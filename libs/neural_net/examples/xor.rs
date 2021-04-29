extern crate neural_net;

use neural_net::*;

fn main() {
    let inputs = vec![
        vec![1.0, -1.0, -1.0],
        vec![1.0, -1.0, 1.0],
        vec![1.0, 1.0, -1.0],
        vec![1.0, 1.0, 1.0],
    ];

    let targets = vec![
        F64Vector::from_vec(vec![-1.0]),
        F64Vector::from_vec(vec![1.0]),
        F64Vector::from_vec(vec![1.0]),
        F64Vector::from_vec(vec![-1.0]),
    ];

    let layer_one_weights = vec![
        vec![0.14, -0.07, 0.10],
        vec![0.13, -0.23, 0.12],
        vec![-0.23, 0.11, 0.03],
    ];
    let layer_one_bias = vec![0.0, 0.0, 0.0];
    let layer_two_weights = vec![vec![0.08], vec![0.04], vec![-0.05]];
    let layer_two_bias = vec![0.0];

    let mut net = NeuralNetwork::load_weights(
        vec![
            LayerWeights {
                weights: layer_one_weights,
                bias: layer_one_bias,
            },
            LayerWeights {
                weights: layer_two_weights,
                bias: layer_two_bias,
            },
        ],
        0.05,
        0.2,
    );

    let tolerance = 0.1;
    let num_epocs = 100;

    let mut can_stop = false;

    for i in 0..num_epocs {
        println!("Starting training epoch {}", i);

        for (input_vec, target) in inputs.iter().zip(targets.iter()) {
            let out = net.propagate_vec(input_vec.to_vec());
            let diff = target - &out;

            if diff[0].abs() >= tolerance {
                can_stop = true;
            }

            println!("Absolute Error at iteration {} is {}", i, diff[0].abs());
            net.backprop(diff);
        }
    }

    if !can_stop {
        println!("Sad did not learn XOR :(")
    }
}
