mod activation_funcs;

/// A network of neurons
pub struct NeuralNetwork { 

    // the implementation of the layer does not matter. 
    // Expert neural-network packages create a set of optimal matrix operations 
    // that severely speed up the 
    layers: Vec<Box<dyn Layer>>
} 

/// A layer topology is simply the number of neurons per a single layer.
#[derive(Copy, Clone, Debug)]
pub struct LayerTopology { 
    num_neurons : usize
}

/// A trait for the behavior of what a nueral net layer should be like
trait Layer { 
    fn propagate(&self, inputs: &Vec<f64>) -> Vec<f64> ;
}

struct NeuronVectorLayer { 
    neurons: Vec<Neuron>
} 

impl Layer for NeuronVectorLayer { 

    /// propagates the input to each neuron in a layer.
    /// Outputs a vector with the output of each neuron propagating on the whole input
    /// Where the ith value is the response of the ith-neuron
    fn propagate(&self, inputs: &Vec<f64>) -> Vec<f64> { 
        self.neurons.iter()
                    .map(|neuron| neuron.propagate(&inputs))
                    .collect()
    }
}

// A neuron keeps track of an internal bias and the set of weights
// it has relative to 
struct Neuron { 
    bias : f64, 
    weights : Vec<f64>,
}

impl Neuron {

    /// Takes an input and applies the neuron's weights to it
    fn propagate(&self, inputs: &Vec<f64>) -> f64 { 
        
        assert_eq!(self.weights.len(), inputs.len());
        // apply each of the weights and apply it to its respective input
        let weighted_sum = self.weights.iter()
            .zip(inputs.iter())
            .map(|(weight, input)| weight * input)
            .sum::<f64>();

        // this neuron will assume to always use the ReLU activation function for now
        (self.bias + weighted_sum).max(0.0)
    }
}

impl NeuralNetwork { 

    /// propagates forward the input throughout the network and outputs the output from the 
    /// final layer
    pub fn propagate(&self, inputs : Vec<f64>) -> Vec<f64> { 
        self.layers
        .iter()
        .fold(inputs, |next_inputs, layer| layer.propagate(&next_inputs) )
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
    pub fn random(topology: &Vec<LayerTopology>) -> Self { 
        todo!()
    }

    /// creates a brand new network using pre-determined weights given 
    pub fn load_weights(weights : &Vec<&[&[f64]]>) -> Self {
        todo!()
     }
}



#[cfg(test)]
mod tests {

    #[test]
    fn new_random_test() { 

    }

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
