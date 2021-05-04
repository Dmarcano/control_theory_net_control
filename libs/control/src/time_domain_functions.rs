/// returns
///
/// $ 1/20 - (exp(-5*t)*(exp(5^(1/2)*t)/2 + exp(-5^(1/2)*t)/2 + 5^(1/2)*(exp(5^(1/2)*t)/2 - exp(-5^(1/2)*t)/2)))/20 $
///
/// $ 1/20 - 0.5*e^(-5*t)*e^(5^(1/2)*t) + e^(-5^(1/2)) $
pub fn plant(t: f64) -> f64 {
    let out = 1.0 / 20.0 - 0.5 * ((-5.0 * t).exp()) * (5.0_f64.sqrt() * t).exp();
    todo!()
}

/// sqrt(15) exp(- 5 t - sqrt(15) t) (exp(2 sqrt(15) t) - 1)
/// /
/// 30
pub fn open_loop_impulse_plant(t: f64) -> f64 {
    let root_15 = 15.0_f64.sqrt();
    let out = root_15 * (-5.0 * t - root_15).exp();
    todo!()
}
