#[derive(Debug)]
pub enum ActivationFunction {
    ReLu,
    Sigmoid,
    TanH,
    LeakyRelu,
}

pub(crate) struct ReLu;
pub(crate) struct Sigmoid;
pub(crate) struct TanH;
pub(crate) struct LeakyRelu;

pub(crate) fn get_functions(
    func: ActivationFunction,
) -> (&'static dyn Fn(f64) -> f64, &'static dyn Fn(f64) -> f64) {
    match func {
        ActivationFunction::ReLu => return (&ReLu::activation, &ReLu::derivative),
        ActivationFunction::TanH => return (&TanH::activation, &TanH::derivative),
        ActivationFunction::Sigmoid => return (&Sigmoid::activation, &Sigmoid::derivative),
        other => {
            unimplemented!("Still have to implement {:?} activation function", other)
        }
    }
}

impl ReLu {
    pub fn activation(input: f64) -> f64 {
        input.max(0.0)
    }

    pub fn derivative(input: f64) -> f64 {
        match input {
            val if val > 0.0 => 1.0,
            _ => 0.0,
        }
    }
}

impl Sigmoid {
    pub fn activation(input: f64) -> f64 {
        (1.0) / (1.0 + (-1.0 * input.exp()))
    }

    pub fn derivative(input: f64) -> f64 {
        Sigmoid::activation(input) * (1.0 - Sigmoid::activation(input))
    }
}

impl TanH {
    pub fn activation(input: f64) -> f64 {
        input.tanh()
    }

    pub fn derivative(input: f64) -> f64 {
        1.0 - (input.tanh().powf(2.0))
    }
}

#[cfg(test)]
mod tests {
    use crate::activation_funcs::*;

    #[test]
    fn relu_test() {
        let relu = ReLu::activation;
        let in_out_pairs = [(1.0, 1.0), (1.5, 1.5), (0.0, 0.0), (-1.0, 0.0)];

        for (input, expected) in in_out_pairs.iter() {
            let out = ReLu::activation(*input);
            assert!(approx::relative_eq!(out, expected));
        }
    }

    fn relu_deriv_test() {}
}
