use rust_ml_algo::models::logistic_regression::LogisticRegression;

fn main() {
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 0.0];
    let mut model = LogisticRegression::new(2, 0.1);
    model.train(&x, &y, 1000);

    let preds = model.predict(&x);
    println!("Predictions: {:?}", preds);
}

