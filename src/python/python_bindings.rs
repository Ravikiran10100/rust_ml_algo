use pyo3::prelude::*;
use crate::models::linear_regression::LinearRegression;


//Python wrapper around Rust LinearRegression
#[pyclass]
pub struct PyLinearRegression {
    inner: LinearRegression,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    fn new(num_features: usize, learning_rate: f32) -> Self {
        PyLinearRegression { inner: LinearRegression::new(num_features, learning_rate) }
    }

    fn train(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>, epochs: usize) {
        self.inner.train(&x, &y, epochs);
    }

    fn predict(&self, x: Vec<Vec<f32>>) -> Vec<f32> {
        self.inner.predict(&x)
    }
}

