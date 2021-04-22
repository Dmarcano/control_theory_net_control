use crate::{Layer, LayerWeights, F64Vector};
use nalgebra::{Matrix, DMatrix};


struct LayerMatrix { 
    mat : DMatrix<f64>
}


impl Layer for LayerMatrix { 
    
    fn propagate(&self, inputs: &F64Vector) -> F64Vector{ 
        let out = inputs.transpose() * &self.mat;
        out.transpose()
    }

    fn new_random(&self, num_neurons : usize) -> Box<dyn Layer > { 
        todo!() 
    }

    fn new_from_weights(&self, _: LayerWeights) ->Box<dyn Layer> { 
        todo!()
    }

}