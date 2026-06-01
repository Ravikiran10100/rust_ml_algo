use pyo3::prelude::*;

pub mod models;
pub mod python;

use python::linear_regression::PyLinearRegression;

#[pymodule]
fn rust_ml(
    m: &Bound<'_, PyModule>
) -> PyResult<()> {
    m.add_class::<PyLinearRegression>()?;
    Ok(())
}
