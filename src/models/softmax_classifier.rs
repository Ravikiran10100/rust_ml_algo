pub struct SoftmaxClassfier {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub learning_rate: f32,
}

impl SoftmaxClassfier {
    pub fn new(num_feature: usize, num_classes: usize, learning_rate: f32) -> Self {
        SoftmaxClassfier {
            weights: vec![vec![0.0; num_classes]; num_feature],
            bias: vec![0.0; num_classes],
            learning_rate,
        }
    }

    //Softmax function with numerical stability
    fn softmax(&self, logits: &Vec<f32>) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_vals: Vec<f32> = logits.iter().map(|z| (z - max_logit).exp()).collect();

        let sum_exp: f32 = exp_vals.iter().sum();

        exp_vals.iter().map(|e| e / sum_exp).collect()
    }

    //Forward Pass: compute probabilities
    pub fn predict(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        x.iter().map(|row| {
            let mut logits = vec![0.0; self.bias.len()];

            for c in 0..self.bias.len() {
                let mut z = self.bias[c];
                for f in 0..row.len() {
                    z += row[f] * self.weights[f][c];
                }
                logits[c] = z;
            }
            //Convert logits -> Probabilities
            self.softmax(&logits)
        })
        .collect()
    }

    //Train using manual gradient descent
    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<usize>, epochs: usize) {
        let n_samples = y.len() as f32;
        let num_classes = self.bias.len();

        for epoch in 0..epochs {
            let predictions = self.predict(x);

            //Initialize gradients
            let mut grad_w = vec![vec![0.0; num_classes]; self.weights.len()];
            let mut grad_b = vec![0.0; num_classes];

            for i in 0..x.len() {
                let probs = &predictions[i];

                //one-hot encoding impplicitly handled
                for c in 0..num_classes {
                    let target = if y[i] == c { 1.0 } else { 0.0 };
                    let error = probs[c] - target;

                    for f in 0..self.weights.len() {
                        grad_w[f][c] += error * x[i][f];
                    }

                    grad_b[c] += error;
                }
            }

            //update weights
            for f in 0..self.weights.len() {
                for c in 0..num_classes {
                    self.weights[f][c] -= self.learning_rate * grad_w[f][c] / n_samples;
                }
            }

            //update bias
            for c in 0..num_classes {
                self.bias[c] -= self.learning_rate * grad_b[c] / n_samples;
            }

            //compute loss
            if epoch % 100 == 0 {
                let mut loss = 0.0;

                for i in 0..y.len() {
                    let prob = predictions[i][y[i]].max(1e-7);
                    loss += -prob.ln();
                }

                loss /= n_samples;
                println!("Epoch {}: CrossEntropy Loss = {:.4}", epoch, loss);
            }
        }
    }
}