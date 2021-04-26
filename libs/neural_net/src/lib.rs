mod activation_funcs;
mod network_impl;
use nalgebra::{DVector, RowDVector};
use rand::{rngs::OsRng, RngCore};

use network_impl::matrix_impl::LayerMatrix;

/// A double precision dynamic vector for use with neural nets
pub type F64Vector = RowDVector<f64>;

/// A network of neurons
pub struct NeuralNetwork {
    // the implementation of the layer does not matter.
    // Expert neural-network packages create a set of optimal matrix operations
    // that severely speed up the operations so we add a trait object to signify that we may use
    // multiple implementations of such object
    layers: Vec<LayerMatrix>,
}

/// A layer topology is simply the number of neurons per a single layer.
#[derive(Copy, Clone, Debug)]
pub struct LayerTopology {
    /// The number of neurons in one layer
    num_neurons: usize,
}

pub struct LayerWeights {
    /// A matrix of layer weights where each row is a neuron of the current layer and the columns are the weights to neurons in
    /// the subsequent layer
    pub weights: Vec<Vec<f64>>,
}

/// A trait for the behavior of what a nueral net layer should be like
trait Layer {
    /// propagates the input to each neuron in a layer.
    /// Outputs a vector with the output of each neuron propagating on the whole input
    /// Where the ith value is the response of the ith-neuron
    fn propagate(&self, inputs: &F64Vector) -> F64Vector;

    fn get_inner_repr<'a>(&'a self) -> Box<dyn Iterator<Item = &f64> + 'a>;
}

impl NeuralNetwork {
    /// propagates forward the input throughout the network and outputs the output from the
    /// final layer
    pub fn propagate_vec(&self, inputs: Vec<f64>) -> F64Vector {
        let transform = RowDVector::from_vec(inputs);

        self.layers.iter().fold(transform, |next_inputs, layer| {
            layer.propagate(&next_inputs)
        })
    }

    /// propagates forward the input throughout the network and outputs the output from the
    /// final layer
    pub fn propagate(&self, inputs: F64Vector) -> F64Vector {
        self.layers
            .iter()
            .fold(inputs, |next_inputs, layer| layer.propagate(&next_inputs))
    }

    /// creates a brand new network using random weights and biases
    ///
    /// ## Arguments
    /// * topology - A vector of LayerToplogies which outlines the length of eahc layer in the network
    ///
    /// ## Example
    /// ```
    /// let topology = vec![
    ///  LayerTopology{4},
    ///  LayerTopology{3},
    ///  LayerTopology{1}];
    ///
    /// let net = NeuralNetwork::random(topology);
    /// ```
    ///
    /// The example above creates a three layer neural net with 4 input neurons, 3 neurons in a hidden layer and a single output neuron. 
    /// 
    /// ## Note on Bias 
    /// The network does not automatically add a bias neuron and leaves
    /// it to users to augment input vectors to include bias terms in their dataset
    ///
    pub fn random(topology: &[LayerTopology]) -> Self {
        // we want networks of at least 1 hidden layer
        assert!(topology.len() > 1);
        let mut rng = OsRng;

        let layers: Vec<LayerMatrix> = topology
            .windows(2)
            .map(|adjacent_layers| {
                let layer: LayerMatrix = LayerMatrix::new_random(
                    adjacent_layers[0].num_neurons,
                    adjacent_layers[1].num_neurons,
                    &mut rng,
                );
                layer
            })
            .collect();

        NeuralNetwork { layers }
    }

    /// creates a brand new network using pre-determined weights given
    /// It assumes that the bias weight of each layer is included in the network topology, that is
    /// the neuron's individual bias are all summed to one bias neuron included in the input weights
    pub fn load_weights(weights: Vec<LayerWeights>) -> Self {
        let mut built_layers = Vec::new();

        for layer_weights in weights {
            let layer = LayerMatrix::new_from_weights(layer_weights);
            built_layers.push(layer);
        }
        NeuralNetwork {
            layers: built_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn new_random_test() {}

    #[test]
    fn new_from_weights_test() {
        // unimplemented!();
    }

    #[test]
    fn propagation_test() {

        // normal prop
        let layer_one_weights = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let layer_two_weights = vec![vec![0.5], vec![0.1]]; 
        let layer_three_weights = vec![vec![1.0]]; 

        let net = NeuralNetwork::load_weights(
            vec![
                LayerWeights{weights : layer_one_weights}, 
                LayerWeights{weights : layer_two_weights}, 
                LayerWeights{weights : layer_three_weights}
            ]
        );

        let input = RowDVector::from_vec(vec![-0.5, 1.5, 2.0]); 
        let expected = 8.7;
        
        let out = net.propagate(input); 
        
        assert_eq!(out[0], expected); 
    }
}
