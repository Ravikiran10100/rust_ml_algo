pub struct KMeans {
    pub centroids: Vec<Vec<f32>>,
    pub k: usize,
    pub max_iters: usize,
}

impl KMeans {
    //Constructor
    pub fn new(k: usize, max_iters: usize) -> Self {
        KMeans {
            centroids: Vec::new(),
            k,
            max_iters,
        }
    }

    // Compute Euclidean distance between two points
    fn distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }

    //Initialize centroids (simple: first k points)
    fn init_centroids(&mut self, x: &Vec<Vec<f32>>) {
        self.centroids = x[..self.k].to_vec();
    }

    //Assign each point to nearest centroid
    fn assign_clusters(&self, x: &Vec<Vec<f32>>) -> Vec<usize> {
        x.iter().map(|point| {
            let mut min_dist = f32::MAX;
            let mut cluster = 0;

            for (i, centroid) in self.centroids.iter().enumerate() {
                let dist = Self::distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    cluster = i;
                }
            }
            cluster
        })
        .collect()
    }

    //Update centroids as mean of assigned points
    fn update_centroids(&mut self, x: &Vec<Vec<f32>>, labels: &Vec<usize>) {
        let dim = x[0].len();

        let mut new_centroids = vec![vec![0.0; dim]; self.k];
        let mut counts = vec![0; self.k];

        for (i, point) in x.iter().enumerate() {
            let cluster = labels[i];
            counts[cluster] += 1;

            for d in 0..dim {
                new_centroids[cluster][d] += point[d];
            }
        }

        for k in 0..self.k {
            if counts[k] > 0 {
                for d in 0..dim {
                    new_centroids[k][d] /= counts[k] as f32;
                }
            }
        }

        self.centroids = new_centroids;
    }

    //Train K-Means
    pub fn fit(&mut self, x: &Vec<Vec<f32>>) -> Vec<usize> {
        self.init_centroids(x);

        let mut labels = vec![0; x.len()];

        for iter in 0..self.max_iters {
            labels = self.assign_clusters(x);

            self.update_centroids(x, &labels);

            println!("Iteration {}: Centroids = {:?}", iter, self.centroids);
        }
        labels
    }
}