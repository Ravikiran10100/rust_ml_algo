pub struct LogisticRegression {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub learning_rate: f32,
}

impl LogisticRegression {
    //Constructor
    pub fn new(num_features: usize, learning_rate: f32) -> Self {
        LogisticRegression {
            weights: vec![0.0; num_features],
            bias: 0.0,
            learning_rate,
        }
    }

    //Sigmoid function
    fn sigmoid(&self, z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    //Forward Pass: Compute probability prediction
    pub fn predict(&self, x: &Vec<Vec<f32>>) -> Vec<f32> {
        x.iter()
            .map(|row| {
                let z: f32 = row
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f32>()
                    + self.bias;
                self.sigmoid(z)
            })
            .collect()
    }

    //Train using manual gradient descent
    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<f32>, epochs: usize) {
        let n_samples = y.len() as f32;

        for epoch in 0..epochs {
            let y_pred = self.predict(x);

            let mut grad_w = vec![0.0; self.weights.len()];
            let mut grad_b = 0.0;

            for i in 0..y.len() {
                let error = y_pred[i] - y[i];
                for j in 0..self.weights.len() {
                    grad_w[j] += error * x[i][j];
                }
                grad_b += error;
            }

            for j in 0..self.weights.len() {
                self.weights[j] -= self.learning_rate * grad_w[j] / n_samples;
            }
            self.bias -= self.learning_rate * grad_b / n_samples;

            if epoch % 100 == 0 {
                let loss = y_pred
                    .iter()
                    .zip(y.iter())
                    .map(|(p, t)| -(t * p.max(1e-7).ln() + (1.0 - t) * (1.0 - p).max(1e-7).ln()))
                    .sum::<f32>()
                    / n_samples;
                println!("Epoch {}: BCE Loss = {:?}", epoch, loss);
            }
        }
    }
}
