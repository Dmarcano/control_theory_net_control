use crate::{F64Vector, Layer, LayerWeights};
use nalgebra::{DMatrix, DVector, RowDVector};
use rand::Rng;

pub struct LayerMatrix {
    mat: DMatrix<f64>,
    activation_func: Box<dyn Fn(f64) -> f64>,
}

impl LayerMatrix {
    pub fn new_from_weights(input: LayerWeights) -> Self {
        assert!(input.weights.len() > 0);
        // iterate over the vectors as rows and use them to create a matrix by row order
        let row_vecs: Vec<RowDVector<f64>> = input
            .weights
            .into_iter()
            .map(|row| RowDVector::from_vec(row))
            .collect();
        let mat = DMatrix::from_rows(row_vecs.as_ref());
        LayerMatrix {
            mat,
            activation_func: Box::new(re_lu),
        }
    }

    // the number of input neurons is the number of rows of the matrix.
    // the number of output neurons is the number of columns inside of the matrix.
    pub fn new_random(
        input_neurons: usize,
        output_neurons: usize,
        rng: &mut dyn rand::RngCore,
    ) -> Self {
        let weights: Vec<f64> = (0..input_neurons * output_neurons)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        let mat = DMatrix::from_vec(input_neurons, output_neurons, weights);

        LayerMatrix {
            mat,
            activation_func: Box::new(re_lu),
        }
    }
}

fn re_lu(val: f64) -> f64 {
    (val).max(0.0)
}

impl Layer for LayerMatrix {
    fn propagate(&self, inputs: &F64Vector) -> F64Vector {
        let out = inputs * &self.mat;

        RowDVector::from_iterator(
            out.len(),
            out.iter().map(|val| (self.activation_func)(*val)),
        )
    }

    fn get_inner_repr<'a>(&'a self) -> Box<dyn Iterator<Item = &f64> + 'a> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::{Layer, LayerMatrix};
    use crate::LayerWeights;
    use approx::relative_eq;
    use nalgebra::RowDVector;

    #[test]
    fn new_random_test() {}

    #[test]
    fn new_from_weights_test() {
        let weights = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let layer = LayerMatrix::new_from_weights(LayerWeights { weights });

        let expected = nalgebra::Matrix2x3::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);

        // iterate and zip => (zip means to iterate two iterators with each item in the same order)
        layer
            .mat
            .as_slice()
            .iter()
            .zip(expected.as_slice().iter())
            .for_each(|(lhs, rhs)| assert!(relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"
    }

    #[test]
    fn propagation_test() {
        let input = RowDVector::from_vec(vec![5.0, 1.0]);
        let weights = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let layer = LayerMatrix::new_from_weights(LayerWeights { weights });

        let out = layer.propagate(&input);
        let expected = vec![0.9, 1.5, 2.1];

        out.as_slice()
            .iter()
            .zip(expected.as_slice().iter())
            .for_each(|(lhs, rhs)| assert!(relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"
    }
}
