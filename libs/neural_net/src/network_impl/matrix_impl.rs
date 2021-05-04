use crate::{F64Vector, LayerWeights};
use nalgebra::{DMatrix, RowDVector};
use rand::Rng;

pub struct LayerMatrix {
    pub mat: DMatrix<f64>,
    pub bias: RowDVector<f64>,
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
        let bias = RowDVector::from_vec(input.bias);

        LayerMatrix {
            mat,
            bias,
        }
    }

    /// computes the
    pub fn weighted_sum(&self, inputs: &F64Vector) -> F64Vector {
        let out = inputs * &self.mat;
        out + &self.bias
    }

    // the number of input neurons is the number of rows of the matrix.
    // the number of output neurons is the number of columns inside of the matrix.
    pub fn new_random(
        input_neurons: usize,
        output_neurons: usize,
        rng: &mut dyn rand::RngCore,
    ) -> Self {
        let weights = (0..input_neurons * output_neurons)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        let bias = RowDVector::from_vec(
            (0..output_neurons)
                .map(|_| rng.gen_range(-1.0..=1.0))
                .collect(),
        );

        let mat = DMatrix::from_vec(input_neurons, output_neurons, weights);

        LayerMatrix {
            mat,
            bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LayerMatrix};
    use crate::LayerWeights;
    use approx::relative_eq;
    use nalgebra::RowDVector;

    #[test]
    fn new_from_weights_test() {
        let weights = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let bias = vec![0.0, 0.0, 0.0];

        let layer = LayerMatrix::new_from_weights(LayerWeights { weights, bias });

        let expected_mat = nalgebra::Matrix2x3::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);

        let expected_bias = vec![0.0, 0.0, 0.0];

        // iterate and zip => (zip means to iterate two iterators with each item in the same order)
        layer
            .mat
            .as_slice()
            .iter()
            .zip(expected_mat.as_slice().iter())
            .for_each(|(lhs, rhs)| assert!(relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"

        layer
            .bias
            .as_slice()
            .iter()
            .zip(expected_bias.iter())
            .for_each(|(lhs, rhs)| assert!(relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"
    }

    #[test]
    fn propagation_test() {
        let input = RowDVector::from_vec(vec![5.0, 1.0]);
        let weights = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let bias = vec![0.0, 0.0, 0.0];

        let layer = LayerMatrix::new_from_weights(LayerWeights { weights, bias });

        let out = layer.weighted_sum(&input);
        let expected = vec![0.9, 1.5, 2.1];

        out.as_slice()
            .iter()
            .zip(expected.as_slice().iter())
            .for_each(|(lhs, rhs)| assert!(relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"
    }

    #[test]
    fn bias_weighted_sum_test() {
        let input = RowDVector::from_vec(vec![5.0, 1.0]);
        let weights = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let bias = vec![1.0, 0.5, 0.0];

        let layer = LayerMatrix::new_from_weights(LayerWeights { weights, bias });

        let out = layer.weighted_sum(&input);
        let expected = vec![1.9, 2.0, 2.1];

        out.as_slice()
            .iter()
            .zip(expected.as_slice().iter())
            .for_each(|(lhs, rhs)| assert!(relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"
    }
}
