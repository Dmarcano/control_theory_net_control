use crate::{F64Vector, Layer, LayerWeights};
use nalgebra::{DMatrix, DVector, RowDVector};

struct LayerMatrix {
    mat: DMatrix<f64>,
    activation_func: Box<dyn Fn(f64) -> f64>,
}

impl LayerMatrix {
    fn new_from_weights(input: LayerWeights) -> Self {
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
            activation_func: Box::new(ReLu),
        }
    }
}

fn ReLu(val: f64) -> f64 {
    (val).max(0.0)
}

impl Layer for LayerMatrix {
    fn propagate(&self, inputs: &F64Vector) -> F64Vector {
        let out = inputs.transpose() * &self.mat;

        DVector::from_iterator(
            out.len(),
            out.transpose()
                .iter()
                .map(|val| (self.activation_func)(*val)),
        )
    }

    fn new_random(&self, num_neurons: usize) -> Box<dyn Layer> {
        todo!()
    }

    // given a set of weights as 2-D vectors we can immediately convert them to our matrix
    fn new_from_weights(&self, input: LayerWeights) -> Box<dyn Layer> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::{Layer, LayerMatrix};
    use crate::LayerWeights;
    use approx::relative_eq;
    use nalgebra::storage::Storage;
    use nalgebra::DVector;

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
        let input = DVector::from_vec(vec![5.0, 10.0]);
        let weights = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let layer = LayerMatrix::new_from_weights(LayerWeights { weights });

        let out = layer.propagate(&input);
        // let expected = vec![];

        // out.as_slice()
        //     .iter()
        //     .zip(expected.as_slice().iter())
        //     .for_each(|(lhs, rhs)| assert!(relative_eq!(lhs, rhs))); // compare "left-hand-side" to "right-hand-side"
    }
}
