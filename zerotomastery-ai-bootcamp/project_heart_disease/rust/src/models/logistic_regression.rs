use std::marker::PhantomData;

use anyhow::Result;
use linfa::prelude::*;
use linfa_linear::{Link, TweedieRegressor};
use ndarray::{Array1, Array2};

use super::{NonTrained, NonTrainedModel, TrainStatus, Trained, TrainedModel};

pub struct LogisticRegressionModel<S: TrainStatus> {
  alpha: f64,
  max_iter: usize,
  tol: f64,
  model: Option<TweedieRegressor<f64>>,
  _marker: PhantomData<S>,
}

impl LogisticRegressionModel<NonTrained> {
  pub fn new() -> Self {
    Self {
      alpha: 0.0,
      max_iter: 100,
      tol: 1e-4,
      model: None,
      _marker: Default::default(),
    }
  }

  pub fn with_alpha(mut self, alpha: f64) -> Self {
    self.alpha = alpha;
    self
  }

  pub fn with_max_iter(mut self, max_iter: usize) -> Self {
    self.max_iter = max_iter;
    self
  }
}

impl NonTrainedModel for LogisticRegressionModel<NonTrained> {
  fn train(self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<Box<dyn TrainedModel>> {
    let dataset = Dataset::new(x_train.clone(), y_train.clone());

    let model = TweedieRegressor::params()
      .power(0.0) // Normal distribution for logistic regression
      .alpha(self.alpha)
      .max_iter(self.max_iter)
      .tol(self.tol)
      .link(Link::Logit) // Logit link for logistic regression
      .fit(&dataset)?;

    Ok(Box::new(LogisticRegressionModel {
      alpha: self.alpha,
      max_iter: self.max_iter,
      tol: self.tol,
      model: Some(model),
      _marker: PhantomData::default(),
    }))
  }
}

impl TrainedModel for LogisticRegressionModel<Trained> {
  fn predict(&self, x_test: &Array2<f64>) -> Result<Array1<f64>> {
    let mut predictions = Array1::zeros(x_test.nrows());
    self
      .model
      .as_ref()
      .unwrap()
      .predict_inplace(x_test, &mut predictions);

    // Convert probabilities to binary predictions (0 or 1)
    for pred in predictions.iter_mut() {
      *pred = if *pred > 0.5 { 1.0 } else { 0.0 };
    }

    Ok(predictions)
  }

  fn predict_proba(&self, x_test: &Array2<f64>) -> Result<Array2<f64>> {
    let mut probabilities = Array1::zeros(x_test.nrows());
    self
      .model
      .as_ref()
      .unwrap()
      .predict_inplace(x_test, &mut probabilities);

    // Convert to 2D array with two columns (class 0 and class 1 probabilities)
    let mut proba_2d = Array2::zeros((x_test.nrows(), 2));
    for (i, &prob) in probabilities.iter().enumerate() {
      proba_2d[(i, 0)] = 1.0 - prob; // Probability of class 0
      proba_2d[(i, 1)] = prob; // Probability of class 1
    }

    Ok(proba_2d)
  }
}
