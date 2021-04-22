mod activation_funcs;
mod network_impl; 



/// A network of neurons
pub struct NeuralNetwork {
    // the implementation of the layer does not matter.
    // Expert neural-network packages create a set of optimal matrix operations
    // that severely speed up the
    layers: Vec<Box<dyn Layer>>,
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
    fn propagate(&self, inputs: &Vec<f64>) -> Vec<f64>;

    fn new_random(&self) -> Box<dyn Layer>; 

    fn new_from_weights(&self, weights : LayerWeights) -> Box<dyn Layer>;
}

impl NeuralNetwork {
    /// propagates forward the input throughout the network and outputs the output from the
    /// final layer
    pub fn propagate(&self, inputs: Vec<f64>) -> Vec<f64> {
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
    /// The example above creates a three layer neural net with 4 input neurons, 3 neurons in a hidden layer and a single output neuron
    ///
    pub fn random(topology: &Vec<LayerTopology>) -> Self {
        todo!()
    }

    /// creates a brand new network using pre-determined weights given
    /// It assumes that the bias weight of each layer is included in the network topology, that is
    /// the neuron's individual bias are all summed to one bias neuron included in the input weights
    pub fn load_weights(weights: Vec<LayerWeights>) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn new_random_test() {}

    #[test]
    fn new_from_weights_test() {
        unimplemented!();
    }

    #[test]
    fn propagation_test() {
        unimplemented!();
    }

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
