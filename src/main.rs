use rust_ml_algo::models::pca::PCA;

fn main() {
    let x = vec![
        vec![2.5, 2.4],
        vec![0.5, 0.7],
        vec![2.2, 2.9],
        vec![1.9, 2.2],
        vec![3.1, 3.0]
    ];

    let mut pca = PCA::new(1);
    pca.fit(&x);

    let transformed = pca.transform(&x);

    println!("Reduced Data: {:?}", transformed)
}