pub enum ActivationFunctions {
    ReLU(ActivationFunction),
    Sigmoid(ActivationFunction),
    TanH(ActivationFunction),
}

pub struct ActivationFunction { 

    function : Box<dyn Fn(f64) -> f64>,
    derivative : Box<dyn Fn(f64) -> f64>
}
