use core::f32;

pub struct PCA {
    pub n_components: usize,
    pub components: Vec<Vec<f32>>,
    pub mean: Vec<f32>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            components: Vec::new(),
            mean: Vec::new(),
        }
    }

    // Compute mean of each feature
    fn compute_mean(&mut self, x: &Vec<Vec<f32>>) {
        let n_samples = x.len();
        let n_features = x[0].len();

        self.mean = vec![0.0; n_features];

        for row in x {
            for j in 0..n_features {
                self.mean[j] += row[j];
            }
        }

        for j in 0..n_features {
            self.mean[j] /= n_samples as f32;
        }
    }

    // Center Data
    fn center_data(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        x.iter().map(|row| {
            row.iter().enumerate().map(|(j, val)| val - self.mean[j]).collect()
        })
        .collect()
    }

    //Compute covariance matrix
    fn covariance_matrix(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let n_samples = x.len();
        let n_features = x[0].len();

        let mut cov = vec![vec![0.0; n_features]; n_features];

        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    cov[i][j] += x[k][i] * x[k][j];
                }
                cov[i][j] /= n_samples as f32;
            }
        }
        cov
    }

    //Power iteration to get top eigenvector
    fn power_iteration(&self, matrix: &Vec<Vec<f32>>, iterations: usize) -> Vec<f32> {
        let n = matrix.len();
        let mut v = vec![1.0; n];

        for _ in 0..iterations {
            let mut new_v = vec![0.0; n];

            for i in 0..n {
                for j in 0..n {
                    new_v[i] += matrix[i][j] * v[j];
                }
            }

            let norm = (new_v.iter().map(|x| x * x).sum::<f32>()).sqrt();

            if norm == 0.0 {
                break;
            }
            
            for i in 0..n {
                new_v[i] /= norm;
            }

            v = new_v;
        }
        v
    }

    //Fit PCA Model
    pub fn fit(&mut self, x: &Vec<Vec<f32>>) {
        self.compute_mean(x);
        
        let centered = self.center_data(x);

        let mut cov = self.covariance_matrix(&centered);

        self.components.clear();

        for _ in 0..self.n_components {
            let eigenvector = self.power_iteration(&cov, 100);
            self.components.push(eigenvector.clone());

            for i in 0..cov.len() {
                for j in 0..cov.len() {
                    cov[i][j] -= eigenvector[i] * eigenvector[j];
                }
            }
        }
    }

    // Transform data to lower dimension
    pub fn transform(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let centered = self.center_data(x);

        centered.iter().map(|row| {
            self.components.iter().map(|comp| {
                row.iter().zip(comp.iter()).map(|(x, c)| x * c).sum()
            }).collect()
        }).collect()
    }
}