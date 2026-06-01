use pyo3::prelude::*;

use crate::models::linear_regression::LinearRegression;

#[pyclass(name = "LinearRegression")]
pub struct PyLinearRegression {
    model: Option<LinearRegression>,
    learning_rate: f32,
    epochs: usize,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    fn new(learning_rate: Option<f32>, epochs: Option<usize>) -> Self {
        Self {
            model: None,
            learning_rate: learning_rate.unwrap_or(0.01),
            epochs: epochs.unwrap_or(1000),
        }
    }

    fn fit(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>) {
        let num_features = x[0].len();

        let mut model = LinearRegression::new(num_features, self.learning_rate);

        model.train(&x, &y, self.epochs);

        self.model = Some(model);
    }

    fn predict(&self, x: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        match &self.model {
            Some(model) => Ok(model.predict(&x)),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model has not been fitted yet",
            )),
        }
    }
}
