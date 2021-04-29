use crate::{F64Vector, Layer, LayerWeights};
use nalgebra::RowDVector;

struct NeuronVectorLayer {
    neurons: Vec<Neuron>,
}

// A neuron keeps track of an internal bias and the set of weights
// it has relative to
struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}

impl NeuronVectorLayer {
    pub fn new_random(
        input_neurons: usize,
        output_neurons: usize,
        rng: &mut dyn rand::RngCore,
    ) -> Self {
        todo!()
    }

    pub fn new_from_weights(input: LayerWeights) -> Self {
        assert!(input.weights.len() > 0);
        // iterate over the vectors as rows and use them to create a matrix by row order
        let row_vecs: Vec<Neuron> = input
            .weights
            .into_iter()
            .map(|row| Neuron::new_from_weights(row))
            .collect();

        todo!()
    }
}

impl Layer for NeuronVectorLayer {
    fn propagate(&self, inputs: &F64Vector) -> F64Vector {
        RowDVector::from_vec(
            self.neurons
                .iter()
                .map(|neuron| neuron.propagate(&inputs))
                .collect(),
        )
    }

    fn get_inner_repr<'a>(&'a self) -> Box<dyn Iterator<Item = &f64>> {
        todo!()
    }

    fn backprop(&mut self, err: f64) -> f64 {
        todo!()
    }
}

impl Neuron {
    /// Takes an input and applies the neuron's weights to it
    fn propagate(&self, inputs: &F64Vector) -> f64 {
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

    pub fn new_from_weights(weights: Vec<f64>) -> Self {
        // assert!(weights.len() > 1)
        todo!()
    }
}
