use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::rng;

pub struct TrainTestSplit {
  pub x_train: Array2<f64>,
  pub x_test: Array2<f64>,
  pub y_train: Array1<f64>,
  pub y_test: Array1<f64>,
}

pub fn train_test_split(
  features: &Array2<f64>,
  targets: &Array1<f64>,
  test_size: f64,
) -> Result<TrainTestSplit> {
  if !(0.0..=1.0).contains(&test_size) {
    return Err(anyhow::anyhow!("test_size must be between 0.0 and 1.0"));
  }

  let n_samples = features.nrows();
  let test_size_usize = (n_samples as f64 * test_size) as usize;
  let train_size_usize = n_samples - test_size_usize;

  let mut indices: Vec<usize> = (0..n_samples).collect();
  indices.shuffle(&mut rng());

  let train_indices = &indices[..train_size_usize];
  let test_indices = &indices[train_size_usize..];

  let x_train = features.select(Axis(0), train_indices);
  let x_test = features.select(Axis(0), test_indices);
  let y_train = targets.select(Axis(0), train_indices);
  let y_test = targets.select(Axis(0), test_indices);

  Ok(TrainTestSplit {
    x_train,
    x_test,
    y_train,
    y_test,
  })
}

pub struct StandardScaler {
  pub mean: Array1<f64>,
  pub std: Array1<f64>,
}

impl StandardScaler {
  pub fn new() -> Self {
    Self {
      mean: Array1::zeros(0),
      std: Array1::zeros(0),
    }
  }

  pub fn fit(&mut self, data: &Array2<f64>) {
    self.mean = data.mean_axis(Axis(0)).unwrap();
    self.std = data.std_axis(Axis(0), 0.0);

    // Avoid division by zero
    for i in 0..self.std.len() {
      if self.std[i] == 0.0 {
        self.std[i] = 1.0;
      }
    }
  }

  pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
    let mut scaled = data.clone();

    for (_i, mut row) in scaled.axis_iter_mut(Axis(0)).enumerate() {
      for j in 0..row.len() {
        row[j] = (row[j] - self.mean[j]) / self.std[j];
      }
    }

    scaled
  }

  pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
    self.fit(data);
    self.transform(data)
  }
}
