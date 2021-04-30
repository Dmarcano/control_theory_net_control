mod activation_funcs;
mod network_impl;
use nalgebra::{DMatrix, RowDVector};
use rand::{rngs::OsRng, Rng};

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

/// A trait for the behavior of what a nueral net layer should be like
trait Layer {
    /// propagates the input to each neuron in a layer.
    /// Outputs a vector with the output of each neuron propagating on the whole input
    /// Where the ith value is the response of the ith-neuron
    fn propagate(&self, inputs: &F64Vector) -> F64Vector;

    fn backprop(&mut self, err: f64) -> f64;

    fn get_inner_repr<'a>(&'a self) -> Box<dyn Iterator<Item = &f64> + 'a>;
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

        let final_out = self.layers.iter().fold(inputs, |next_inputs, layer| {
            let out = layer.weighted_sum(&next_inputs);
            weighted_sums.push(out.clone());
            let activated = out.map(|val| activation_funcs::TanH::activation(val));
            activations.push(activated.clone());
            activated
        });
        // get rid of the final activatino 'output'
        activations.pop();
        self.layer_activations = activations;
        self.layer_weighted_sum = weighted_sums;
        final_out
    }

    pub fn new_backprop(&mut self, err: F64Vector) {
        let mut weight_changes : Vec<DMatrix<f64>> = Vec::with_capacity(self.layers.len());
        let mut bias_changes = Vec::with_capacity(self.layers.len());
        let alpha = self.learning_rate;

        // 1. Computer the gradient of the output layer

        // Compute the derivative of the final weighted sum into the output layer
        let last_deriv = &self.layer_weighted_sum[self.layer_weighted_sum.len() - 1]
            .map(|val| activation_funcs::TanH::derivative(val));

        // take the Hardman product of the error with the derivative of final weighted sum
        let mut prev_delta = err.component_mul(&last_deriv); // we convert to a matrix because of nalgebras type system

        // compute the weight change which is the input activation to the final layer and the delta
        let weight_change =
        &prev_delta * &self.layer_activations[self.layer_activations.len() - 1];

        weight_changes.push(DMatrix::from_columns(&[weight_change.transpose()]));
        bias_changes.push(prev_delta.clone());

        // 2 compute all of the other's gradients
        for ((layer, activation), weighted_sum) in self.layers[1..]
            .iter()
            .zip(self.layer_activations[..&self.layer_activations.len() - 1].iter())
            .zip(self.layer_weighted_sum[..&self.layer_weighted_sum.len() - 1].iter())
            .rev()
        {
            // Take the derivative of the input weighted sum to the layer
            let weighted_sum_deriv =
                weighted_sum.map(|val| activation_funcs::TanH::derivative(val));
            // take the next layer's weights multiplied by that layers error. multiply by the derivative
            // this is this layers delta or error
            let debug = &prev_delta *layer.mat.transpose();
            let delta = debug.component_mul(&weighted_sum_deriv);

            // let debug3 = DMatrix::from_rows(&[activation.clone()]);
            // weight_change = debug2 * debug3;
            let debug2 = delta.transpose();
           let weight_change = debug2 * activation;

            weight_changes.push(weight_change);
            bias_changes.push(delta.clone());
            prev_delta = delta;
        }

        for ((weight_change, bias_change), layer) in weight_changes.iter()
        .zip(bias_changes.iter())
        .zip(self.layers.iter_mut().rev()) { 
            let regulated = weight_change.map(|val| val * alpha);
            let regulated_bias = bias_change.map(|val| val * alpha);
            layer.mat += regulated;
            layer.bias += regulated_bias;
        }
    }

    /// Recomputes network weights by propagation an error vector along with the
    ///
    /// ### Note
    /// Backpropagation works
    pub fn backprop(&mut self, err: F64Vector) {
        // for the output layer the input
        // 1. take the error vector
        // 2. take hadamard product of err by derivative of current layer output => call this delta
        // 3. multiply the input to the current layer (transposed) by delta => call this change_weights
        // 4. add assign the change_weights times learning_rate to the network weights

        // for every single hidden layer
        // 1. Take the previous layers delta
        // 2. multiply it by the previous layer weigts (transpose) => new error vector
        // 3. hadamard product of err and derivative of the output => call this delta
        // 4. multiply the delta by the output of the hidden layer
        //

        let mut weight_changes = Vec::new();
        let alpha = self.learning_rate;

        // t

        let mut prev_delta = err;

        // the first previous layer is a layer of 1's so that the output layer is not
        // changed
        let mut prev_layer_weight = &nalgebra::DMatrix::repeat(
            self.layer_activations[self.layer_activations.len() - 1].nrows(),
            self.layer_activations[self.layer_activations.len() - 1].ncols(),
            1.0,
        );

        // using the index
        for (layer, output) in self
            .layers
            .iter_mut()
            .zip(self.layer_activations.as_slice().windows(2))
            .rev()
        {
            // 1. let the layer err be the transpose previous weights times the previous delta
            let layer_err = &prev_delta * prev_layer_weight.transpose();

            // 2. create the layer output
            let output_deriv = output[1].map(|val| activation_funcs::TanH::derivative(val));

            // 3. use the output derivative and the layer err to find the delta
            let delta = layer_err.component_mul(&output_deriv);

            // the first layer input is its own output

            // 4. use the delta and current layer input to find the weight change necessary
            let debug = &output[0];
            let layer_adjustment = debug.transpose() * &delta; //todo!();

            prev_delta = delta;
            prev_layer_weight = &layer.mat;

            weight_changes.push(layer_adjustment);
        }

        for (weights, weight_change) in self.layers.iter_mut().zip(weight_changes.iter().rev()) {
            let regulated = weight_change.map(|val| val * alpha);
            weights.mat -= regulated;
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
    /// use neural_net::*;
    ///
    /// let topology = vec![
    ///  LayerTopology{num_neurons : 4},
    ///  LayerTopology{num_neurons: 3},
    ///  LayerTopology{num_neurons: 1}];
    ///
    /// //create a network with 3 layers. 4 input neurons, 3 hidden neurons, 1 output neuron
    /// let net = NeuralNetwork::random(&topology, Some(0.1), None);
    /// ```
    ///
    /// The example above creates a three layer neural net with 4 input neurons, 3 neurons in a hidden layer and a single output neuron.
    ///
    /// ## Note on Bias
    /// The network does not automatically add a bias neuron and leaves
    /// it to users to augment input vectors to include bias terms in their dataset
    ///
    pub fn random(topology: &[LayerTopology], alpha: Option<f64>, lambda: Option<f64>) -> Self {
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

        NeuralNetwork {
            layers,
            learning_rate,
            layer_weighted_sum: layer_activations.clone(),
            layer_activations,
            lambda: regularization,
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
    pub fn load_weights(weights: Vec<LayerWeights>, alpha: f64, lambda: f64) -> Self {
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

        // let outputs = vec![F64Vector::from_vec(), len()];
        NeuralNetwork {
            layers: built_layers,
            layer_weighted_sum: layer_activations.clone(),
            layer_activations,
            learning_rate: alpha,
            lambda,
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
        )
    }

    // #[test]
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

    // #[test]
    fn propagation_test() {
        // normal prop
        let mut net = default_network();

        let input = RowDVector::from_vec(vec![-0.5, 1.5, 2.0]);
        let expected = 8.7;

        let out = net.propagate(input);

        assert_eq!(out[0], expected);
    }

    // #[test]
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

    // #[test]
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
        );

        let input = RowDVector::from_vec(vec![2.0, 3.0]);
        let target = RowDVector::from_vec(vec![1.0]);

        let out = net.propagate(input.clone());

        // 1/2 * SSE for the target and output
        let diff = &out - &target;
        let squared_error = squared_err(&target, &out);
        // net.backprop(diff);
        net.new_backprop(diff);

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

        let layer_one_weights = vec![
            vec![0.13, -0.23],
            vec![-0.23, 0.11],
        ];
        let layer_one_bias = vec![0.14, -0.07];
        let layer_two_weights = vec![vec![0.04], vec![0.-0.05]];
        let layer_two_bias = vec![0.08];

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
            0.2,
            0.2,
        );

        let tolerance = 0.1;
        let num_epocs = 200;

        let mut can_stop = false;

        for i in 0..num_epocs {
            println!("Starting training epoch {}\n", i);

            for (input_vec, target) in inputs.iter().zip(targets.iter()) {
                let out = net.propagate_vec(input_vec.to_vec());
                let diff =  target - &out ;

                if diff[0].abs() <= tolerance {
                    can_stop = true;
                }

                println!("Output for input {:?} is {} target = {} diff = {}", input_vec, out[0], target[0], diff[0]);

                // println!("Absolute Error at iteration {} is {}", i, diff[0].abs());
                net.new_backprop(diff);
            }
        }

        if !can_stop {
            println!("Sad did not learn XOR :(")
        }
    }
}
