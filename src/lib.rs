use pyo3::prelude::*;

pub mod models;
pub mod python;

use python::python_bindings::PyLinearRegression;

#[pymodule]
fn rust_ml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLinearRegression>()?;
    Ok(())
}
