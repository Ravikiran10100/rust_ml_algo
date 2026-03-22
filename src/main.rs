use rust_ml_algo::models::linear_regression::LinearRegression;

fn main() {
    let x = vec![vec![1.0, 2.0], vec![2.0, 0.5], vec![3.0, 4.0]];
    let y = vec![12.0, 9.0, 23.0];

    let mut model = LinearRegression::new(2, 0.01);

    model.train(&x, &y, 1000);

    let preds = model.predict(&x);
    println!("Predictions: {:?}", preds);
}