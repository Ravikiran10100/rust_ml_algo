use core::error;

pub struct LinearRegression {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub learning_rate: f32,
}

impl LinearRegression {
    //constructor initialize the weights and bias to zero
    pub fn new(num_features: usize, learning_rate: f32) -> Self {
        LinearRegression {
            weights: vec![0.0; num_features],
            bias: 0.0,
            learning_rate: 0.0,
        }
    }

    //forward pass: compute predictions
    pub fn predict(&self, x: &Vec<Vec<f32>>) -> Vec<f32> {
        x.iter()
            .map(|row| {
                row.iter()
                    .zip(self.weights.iter())
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f32>()
                    + self.bias
            })
            .collect()
    }

    //Train the model using manual gradient decent
    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<f32>, epochs: usize) {
        let n_samples = y.len() as f32;
        for epoch in 0..epochs {
            let y_pred = self.predict(x);

            //compute gradients manually
            let mut grad_w = vec![0.0; self.weights.len()];
            let mut grad_b = 0.0;

            for i in 0..y.len() {
                let error = y_pred[i] - y[i];
                for j in 0..self.weights.len() {
                    grad_w[j] += error * x[i][j];
                }
                grad_b += error;
            }

            //update parameters
            for j in 0..self.weights.len() {
                self.weights[j] -= self.learning_rate * grad_w[j] / n_samples;
            }
            self.bias -= self.learning_rate * grad_b / n_samples;

            //OPTIONAL: Print loss every 100 epochs
            if epoch % 100 == 0 {
                let loss = y_pred
                    .iter()
                    .zip(y.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f32>()
                    / n_samples;
                println!("Epoch {}: Loss = {:.4}", epoch, loss);
            }
        }
    }
}
