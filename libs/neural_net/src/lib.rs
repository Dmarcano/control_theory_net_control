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
    fn propagate(&self, inputs: &Vec<f64>) -> Vec<f64> { 
        todo!()
    }
}

// A neuron keeps track of an internal bias and the set of weights
// it has relative to 
struct Neuron { 
    bias : f64, 
    weights : Vec<f64>,
}

impl Neuron {
    fn propagate(&self, inputs: &Vec<f64>) -> Vec<f64> { 
        todo!()
    }
}

impl NeuralNetwork { 
    pub fn propagate() -> Vec<f64> { 
        unimplemented!("Need to implement propagate")
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
