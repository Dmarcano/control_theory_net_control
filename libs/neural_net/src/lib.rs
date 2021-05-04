pub mod activation_funcs;
mod network_impl;
use nalgebra::{DMatrix, RowDVector};
use rand::{rngs::OsRng, Rng};

use activation_funcs::ActivationFunction;
use network_impl::matrix_impl::LayerMatrix;

/// A double precision dynamic vector for use with neural nets
pub type F64Vector = RowDVector<f64>;

pub fn squared_err(rhs: &F64Vector, lhs: &F64Vector) -> F64Vector {
    // 1/2 * SSE for the target and output
    let diff = rhs - lhs;
    let err = diff.component_mul(&diff);
    err.map(|val| val / 2.0)
}

/// A network of neurons
pub struct NeuralNetwork {
    // the implementation of the layer does not matter.
    // Expert neural-network packages create a set of optimal matrix operations
    // that severely speed up the operations so we add a trait object to signify that we may use
    // multiple implementations of such object
    layers: Vec<LayerMatrix>,

    /// The network outputs that correspond to each laye's weighted output after they have
    /// gone through the activation function
    layer_activations: Vec<F64Vector>,

    layer_weighted_sum: Vec<F64Vector>,

    /// The networks learning rate when training using backpropagation
    pub learning_rate: f64,

    /// The networks regularization parameter. Penalizes large weigth values
    pub lambda: f64,

    activation_func: &'static dyn Fn(f64) -> f64,

    activation_deriv: &'static dyn Fn(f64) -> f64,
}

/// A layer topology is simply the number of neurons per a single layer.
#[derive(Copy, Clone, Debug)]
pub struct LayerTopology {
    /// The number of neurons in one layer
    pub num_neurons: usize,
}

pub struct LayerWeights {
    /// A matrix of layer weights where each row is a neuron of the current layer and the columns are the weights to neurons in
    /// the subsequent layer
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

pub struct BackPropOutput {
    weight_changes: Vec<DMatrix<f64>>,
    bias_changes: Vec<F64Vector>,
}

impl NeuralNetwork {
    /// propagates forward the input throughout the network and outputs the output from the
    /// final layer
    pub fn propagate_vec(&mut self, inputs: Vec<f64>) -> F64Vector {
        let transform = RowDVector::from_vec(inputs);
        self.propagate(transform)
    }

    /// propagates forward the input throughout the network and outputs the output from the
    /// final layer
    pub fn propagate(&mut self, inputs: F64Vector) -> F64Vector {
        let mut weighted_sums = Vec::with_capacity(self.layers.len());
        let mut activations = Vec::with_capacity(self.layers.len());

        activations.push(inputs.clone());

        let activation = self.activation_func;

        let final_out = self.layers.iter().fold(inputs, |next_inputs, layer| {
            let out = layer.weighted_sum(&next_inputs);
            weighted_sums.push(out.clone());
            let activated = out.map(|val| activation(val));
            activations.push(activated.clone());
            activated
        });
        // get rid of the final activatino 'output'
        activations.pop();
        self.layer_activations = activations;
        self.layer_weighted_sum = weighted_sums;
        final_out
    }

    /// Computes the networks gradient relative to a given the derivative of the error with respect to
    /// the networks output activation.
    ///
    /// ### Note
    ///
    /// Backpropagation right now works on the assumption that the given input is part of a
    /// Sum of Squared Errors loss. That is the that the derivative is simply the raw error of the network output
    /// and the target
    pub fn backprop(&mut self, err: F64Vector) -> BackPropOutput {
        let mut weight_changes: Vec<DMatrix<f64>> = Vec::with_capacity(self.layers.len());
        let mut bias_changes = Vec::with_capacity(self.layers.len());

        // 1. Computer the gradient of the output layer
        let deriv = self.activation_deriv;

        // Compute the derivative of the final weighted sum into the output layer
        let last_deriv =
            &self.layer_weighted_sum[self.layer_weighted_sum.len() - 1].map(|val| deriv(val));

        // take the Hardman product of the error with the derivative of final weighted sum
        let mut prev_delta = err.component_mul(&last_deriv); // we convert to a matrix because of nalgebras type system

        // compute the weight change which is the input activation to the final layer and the delta
        let last_activation = &self.layer_activations[self.layer_activations.len() - 1];
        let weight_change = last_activation.transpose() * &prev_delta;

        weight_changes.push(weight_change);
        bias_changes.push(prev_delta.clone());

        // 2 compute all of the other's gradients
        for ((layer, activation), weighted_sum) in self.layers[1..]
            .iter()
            .zip(self.layer_activations[..&self.layer_activations.len() - 1].iter())
            .zip(self.layer_weighted_sum[..&self.layer_weighted_sum.len() - 1].iter())
            .rev()
        {
            // Take the derivative of the input weighted sum to the layer
            let weighted_sum_deriv = weighted_sum.map(|val| deriv(val));
            // take the next layer's weights multiplied by that layers error. multiply by the derivative
            // this is this layers delta or error
            let debug = &layer.mat * &prev_delta.transpose();
            let delta = debug.component_mul(&weighted_sum_deriv.transpose());

            let weight_change = &delta * activation;

            // update our variables to descent
            weight_changes.push(weight_change.transpose());
            bias_changes.push(delta.transpose());
            prev_delta = delta.transpose();
        }
        // reverse the weights to get them in ascending weight order
        weight_changes.reverse();
        bias_changes.reverse();
        BackPropOutput {
            weight_changes,
            bias_changes,
        }
        // self.update_weights(weight_changes, bias_changes);
    }

    /// Combines the left hand side and right hand side backpropagation outputs into the sum of one output
    pub fn combine_backprop_outputs(lhs: BackPropOutput, rhs: &BackPropOutput) -> BackPropOutput {
        let weight_changes = lhs
            .weight_changes
            .into_iter()
            .zip(rhs.weight_changes.iter())
            .map(|(left_weight_changes, right_weight_changes)| {
                left_weight_changes + right_weight_changes
            })
            .collect();

        let bias_changes = lhs
            .bias_changes
            .into_iter()
            .zip(rhs.bias_changes.iter())
            .map(|(left_bias_changes, right_bias_changes)| left_bias_changes + right_bias_changes)
            .collect();

        BackPropOutput {
            weight_changes,
            bias_changes,
        }
    }

    /// Updates a networks weights based on the given backpropagation output and the
    /// networks learning rate
    /// Currently this network does not implement any regularization.
    pub fn update_weights(&mut self, output: &BackPropOutput) {
        let alpha = self.learning_rate;

        for ((weight_change, bias_change), layer) in output
            .weight_changes
            .iter()
            .zip(output.bias_changes.iter())
            .zip(self.layers.iter_mut())
        {
            let regulated = weight_change.map(|val| val * alpha);
            let regulated_bias = bias_change.map(|val| val * alpha);
            layer.mat += regulated;
            layer.bias += regulated_bias;
        }
    }

    /// creates a brand new network using random weights and biases
    ///
    /// ## Arguments
    /// * topology - A vector of LayerToplogies which outlines the length of eahc layer in the network
    /// * alpha - The network's learning rate and weight change on gradient descent. Reccomended to keep between 0 and 1.
    /// * lambda - A regularization parameter. Penalizes large weight values. Defaults to 0 if None is provided.
    ///
    /// ## Example
    /// ```
    /// use neural_net::{NeuralNetwork,LayerTopology, activation_funcs::ActivationFunction };
    ///
    /// let topology = vec![
    ///  LayerTopology{num_neurons : 4},
    ///  LayerTopology{num_neurons: 3},
    ///  LayerTopology{num_neurons: 1}];
    ///
    /// //create a network with 3 layers. 4 input neurons, 3 hidden neurons, 1 output neuron. It has a ReLu activation function
    /// let net = NeuralNetwork::random(&topology, Some(0.1), None, ActivationFunction::ReLu);
    /// ```
    ///
    /// The example above creates a three layer neural net with 4 input neurons, 3 neurons in a hidden layer and a single output neuron.
    ///
    /// ## Note on Bias
    /// The network automatically adds a bias portion to each neuron and does not expect
    /// any implementor to augment its input data to accomadate bias neurons.
    pub fn random(
        topology: &[LayerTopology],
        alpha: Option<f64>,
        lambda: Option<f64>,
        activation_func: ActivationFunction,
    ) -> Self {
        // we want networks of at least 1 hidden layer
        assert!(topology.len() > 1);
        let mut rng = OsRng;
        let mut layer_activations = Vec::with_capacity(topology.len());

        let layers: Vec<LayerMatrix> = topology
            .windows(2)
            .map(|adjacent_layers| {
                let layer: LayerMatrix = LayerMatrix::new_random(
                    adjacent_layers[0].num_neurons,
                    adjacent_layers[1].num_neurons,
                    &mut rng,
                );
                // output neurons for backprop
                let output_vec = F64Vector::from_vec(vec![0.0; adjacent_layers[1].num_neurons]);
                layer_activations.push(output_vec);
                layer
            })
            .collect();

        let learning_rate = match alpha {
            None => rng.gen_range(0.0..=0.2),
            Some(val) => val,
        };
        let regularization = match lambda {
            None => 0.0,
            Some(val) => val,
        };

        let (activation, derivative) = activation_funcs::get_functions(activation_func);

        NeuralNetwork {
            layers,
            learning_rate,
            layer_weighted_sum: layer_activations.clone(),
            layer_activations,
            lambda: regularization,
            activation_deriv: derivative,
            activation_func: activation,
        }
    }

    /// creates a brand new network using pre-determined weights given
    /// It assumes that the bias weight of each layer is included in the network topology, that is
    /// the neuron's individual bias are all summed to one bias neuron included in the input weights
    ///
    /// ## Arguments
    /// * weights: Vector of LayerWeighst that are used to describe the network
    /// * alpha : Learning rate should changes be needed
    /// * lambda : Regularization parameter should changes be needed
    pub fn load_weights(
        weights: Vec<LayerWeights>,
        alpha: f64,
        lambda: f64,
        activation_func: ActivationFunction,
    ) -> Self {
        let mut built_layers = Vec::new();
        let mut layer_activations = Vec::with_capacity(weights.len());

        for layer_weights in weights {
            assert!(
                layer_weights.weights.len() > 0,
                "Expected a non-zero layer weight but was given {}",
                layer_weights.weights.len()
            );
            // each layer will output a 1x(num_cols) output so we create a vector with the length of each column
            let col_length = layer_weights.weights[0].len();
            let output_vec = F64Vector::from_vec(vec![0.0; col_length]);
            layer_activations.push(output_vec);
            // now build the layer from the layer weights
            let layer = LayerMatrix::new_from_weights(layer_weights);
            built_layers.push(layer);
        }

        let (activation, derivative) = activation_funcs::get_functions(activation_func);

        // let outputs = vec![F64Vector::from_vec(), len()];
        NeuralNetwork {
            layers: built_layers,
            layer_weighted_sum: layer_activations.clone(),
            layer_activations,
            learning_rate: alpha,
            lambda,
            activation_deriv: derivative,
            activation_func: activation,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    fn default_network() -> NeuralNetwork {
        let layer_one_weights = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let layer_two_weights = vec![vec![0.5], vec![0.1]];

        let layer_one_bias = vec![0.0, 0.0];

        let layer_two_bias = vec![0.0];

        NeuralNetwork::load_weights(
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
            0.1,
            0.1,
            activation_funcs::ActivationFunction::ReLu,
        )
    }

    #[test]
    // test if our intermediate network outputs are what we expect them to be
    fn output_storage_test() {
        let mut net = default_network();

        let input_one = RowDVector::from_vec(vec![-0.5, 1.5, 2.0]);

        let _ = net.propagate(input_one.clone());
        let expected_output_second = F64Vector::from_vec(vec![14.0, 17.0]); // expected value of the hidden layer

        let expected = vec![input_one, expected_output_second];

        net.layer_activations
            .as_slice()
            .iter()
            .zip(expected.as_slice().iter())
            .for_each(|(lhs, rhs)| assert!(approx::relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"
    }

    #[test]
    fn propagation_test() {
        // normal prop
        let mut net = default_network();

        let input = RowDVector::from_vec(vec![-0.5, 1.5, 2.0]);
        let expected = 8.7;

        let out = net.propagate(input);

        assert_eq!(out[0], expected);
    }

    #[test]
    fn back_prop_test_no_check() {
        let mut net = default_network();

        let input = RowDVector::from_vec(vec![-0.5, 1.5, 2.0]);
        let out = net.propagate(input);

        let target = RowDVector::from_vec(vec![5.0]);

        // 1/2 * SSE for the target and output
        let diff = &target - &out;
        let err = diff.component_mul(&diff);
        let err = err.map(|val| val / 2.0);

        net.backprop(err);
    }

    #[test]
    fn web_backprop_test() {
        let layer_one_weights = vec![vec![0.11, 0.12], vec![0.21, 0.08]];
        let layer_one_bias = vec![0.0, 0.0];
        let layer_two_weights = vec![vec![0.14], vec![0.15]];
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
            0.1,
            activation_funcs::ActivationFunction::ReLu,
        );

        let input = RowDVector::from_vec(vec![2.0, 3.0]);
        let target = RowDVector::from_vec(vec![1.0]);

        let out = net.propagate(input.clone());

        // 1/2 * SSE for the target and output
        let diff = &target - &out;
        let squared_error = squared_err(&target, &out);
        let backprop_out = net.backprop(diff);
        net.update_weights(&backprop_out);

        let second_out = net.propagate(input);

        let second_squared_err = squared_err(&target, &second_out);

        assert!(squared_error > second_squared_err);
    }

    #[test]
    fn xor_test() {
        let inputs = vec![
            vec![-1.0, -1.0],
            vec![-1.0, 1.0],
            vec![1.0, -1.0],
            vec![1.0, 1.0],
        ];

        let targets = vec![
            F64Vector::from_vec(vec![-1.0]),
            F64Vector::from_vec(vec![1.0]),
            F64Vector::from_vec(vec![1.0]),
            F64Vector::from_vec(vec![-1.0]),
        ];

        let layer_one_weights = vec![vec![0.13, -0.23], vec![-0.23, 0.11]];
        let layer_one_bias = vec![0.14, -0.07];
        let layer_two_weights = vec![vec![0.04], vec![0. - 0.05]];
        let layer_two_bias = vec![0.08];

        let mut backprop_accum: Option<BackPropOutput> = None;

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
            0.5,
            0.2,
            ActivationFunction::TanH,
        );

        let tolerance = 0.1;
        let num_epocs = 100;

        let mut can_stop = false;

        for i in 0..num_epocs {
            can_stop = true;

            println!("Starting training epoch {}\n", i);

            for (input_vec, target) in inputs.iter().zip(targets.iter()) {
                let out = net.propagate_vec(input_vec.to_vec());
                let diff = target - &out;

                if diff[0].abs() >= tolerance {
                    can_stop = false;
                }

                println!(
                    "Output for input {:?} is {} target = {} diff = {}",
                    input_vec, out[0], target[0], diff[0]
                );

                // println!("Absolute Error at iteration {} is {}", i, diff[0].abs());
                let back_prop_out = net.backprop(diff);

                backprop_accum = match backprop_accum {
                    None => Some(back_prop_out),
                    Some(accum) => Some(NeuralNetwork::combine_backprop_outputs(
                        accum,
                        &back_prop_out,
                    )),
                };
            }

            let update = backprop_accum.take().unwrap();
            net.update_weights(&update);

            if can_stop {
                println!("All correct so stopping!");
                break;
            }
        }

        if !can_stop {
            println!("Sad did not learn XOR :(")
        }
    }
}
