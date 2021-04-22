

struct NeuronVectorLayer {
    neurons: Vec<Neuron>,
}


// A neuron keeps track of an internal bias and the set of weights
// it has relative to
struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}


impl Layer for NeuronVectorLayer {
    /// propagates the input to each neuron in a layer.
    /// Outputs a vector with the output of each neuron propagating on the whole input
    /// Where the ith value is the response of the ith-neuron
    fn propagate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}


impl Neuron {
    /// Takes an input and applies the neuron's weights to it
    fn propagate(&self, inputs: &Vec<f64>) -> f64 {
        assert_eq!(self.weights.len(), inputs.len());
        // apply each of the weights and apply it to its respective input
        let weighted_sum = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(weight, input)| weight * input)
            .sum::<f64>();

        // this neuron will assume to always use the ReLU activation function for now
        (self.bias + weighted_sum).max(0.0)
    }
}