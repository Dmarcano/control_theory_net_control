

pub trait ActivationFunction {
    fn activation(&self, input: f64) -> f64;
    fn derivative(&self, input: f64) -> f64;
}

pub struct ReLu;
pub struct Sigmoid;
pub struct TanH;

impl ActivationFunction for ReLu {
    
    fn activation(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    fn derivative(&self, input: f64) -> f64 {
        match input {
            val if val > 0.0 => 1.0,
            _ => 0.0,
        }
    }
}


#[cfg(test)]
mod tests { 
    use crate::activation_funcs::*; 

    #[test]
    fn relu_test() {
        let relu = ReLu{};
        let in_out_pairs = [(1.0, 1.0), (1.5, 1.5), (0.0, 0.0), (-1.0, 0.0)];

        for (input, expected) in in_out_pairs.iter() { 
            let out = relu.activation(*input); 
            assert!(approx::relative_eq!(out, expected));
        }
    }
}
