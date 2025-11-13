use anyhow::Result;
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use rand::Rng;
use crate::models::{NonTrainedModel, TrainedModel};

pub struct RandomForestModel {
  n_trees: usize,
  max_depth: Option<usize>,
  min_samples_split: usize,
  min_samples_leaf: usize,
  max_features: Option<usize>,
}

impl RandomForestModel {
  pub fn new(
    n_trees: Option<usize>,
    max_depth: Option<usize>,
    min_samples_split: Option<usize>,
    min_samples_leaf: Option<usize>,
    max_features: Option<usize>,
  ) -> Self {
    Self {
      n_trees: n_trees.unwrap_or(100),
      max_depth,
      min_samples_split: min_samples_split.unwrap_or(2),
      min_samples_leaf: min_samples_leaf.unwrap_or(1),
      max_features,
    }
  }
}

impl NonTrainedModel for RandomForestModel {
  fn train(self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<Box<dyn TrainedModel>> {
    let mut trees = Vec::new();
    let n_samples = x_train.nrows();
    let n_features = x_train.ncols();
    let max_features = self.max_features.unwrap_or((n_features as f64).sqrt() as usize);
    
    let mut rng = rand::rng();

    for _ in 0..self.n_trees {
      // Bootstrap sampling (sample with replacement)
      let bootstrap_indices: Vec<usize> = (0..n_samples)
        .map(|_| rng.random_range(0..n_samples))
        .collect();
      
      // Random feature selection
      let mut feature_indices: Vec<usize> = (0..n_features).collect();
      // Manual shuffle using Fisher-Yates algorithm
      for i in (1..feature_indices.len()).rev() {
        let j = rng.random_range(0..=i);
        feature_indices.swap(i, j);
      }
      feature_indices.truncate(max_features);
      
      // Create bootstrap dataset
      let x_bootstrap = Array2::from_shape_fn((bootstrap_indices.len(), max_features), |(i, j)| {
        x_train[(bootstrap_indices[i], feature_indices[j])]
      });
      
      let y_bootstrap = Array1::from_shape_fn(bootstrap_indices.len(), |i| {
        y_train[bootstrap_indices[i]] as usize
      });
      
      // Train decision tree
      let dataset = Dataset::new(x_bootstrap, y_bootstrap);
      let tree = DecisionTree::params()
        .max_depth(self.max_depth)
        .fit(&dataset)?;
      
      trees.push((tree, feature_indices));
    }

    Ok(Box::new(FittedRandomForest { trees }))
  }
}

pub struct FittedRandomForest {
  trees: Vec<(DecisionTree<f64, usize>, Vec<usize>)>,
}

impl TrainedModel for FittedRandomForest {
  fn predict(&self, x_test: &Array2<f64>) -> Result<Array1<f64>> {
    let n_samples = x_test.nrows();
    let mut predictions = Vec::with_capacity(n_samples);

    for sample_idx in 0..n_samples {
      let mut votes = Vec::new();
      
      for (tree, feature_indices) in &self.trees {
        // Extract features for this tree
        let sample_features = Array2::from_shape_fn((1, feature_indices.len()), |(_, j)| {
          x_test[(sample_idx, feature_indices[j])]
        });
        
        // Predict with this tree
        let mut prediction = Array1::zeros(1);
        tree.predict_inplace(&sample_features, &mut prediction);
        votes.push(prediction[0]);
      }
      
      // Majority voting
      let vote_counts = votes.iter().fold(std::collections::HashMap::new(), |mut counts, &vote| {
        *counts.entry(vote).or_insert(0) += 1;
        counts
      });
      
      let majority_vote = vote_counts.iter()
        .max_by_key(|&(_, &count)| count)
        .map(|(&vote, _)| vote)
        .unwrap_or(0);
      
      predictions.push(majority_vote as f64);
    }

    Ok(Array1::from_vec(predictions))
  }

  fn predict_proba(&self, x_test: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = x_test.nrows();
    let mut probabilities = Array2::zeros((n_samples, 2));

    for sample_idx in 0..n_samples {
      let mut votes = Vec::new();
      
      for (tree, feature_indices) in &self.trees {
        let sample_features = Array2::from_shape_fn((1, feature_indices.len()), |(_, j)| {
          x_test[(sample_idx, feature_indices[j])]
        });
        
        let mut prediction = Array1::zeros(1);
        tree.predict_inplace(&sample_features, &mut prediction);
        votes.push(prediction[0]);
      }
      
      let class_0_votes = votes.iter().filter(|&&v| v == 0).count() as f64;
      let class_1_votes = votes.iter().filter(|&&v| v == 1).count() as f64;
      let total_votes = votes.len() as f64;
      
      probabilities[(sample_idx, 0)] = class_0_votes / total_votes;
      probabilities[(sample_idx, 1)] = class_1_votes / total_votes;
    }

    Ok(probabilities)
  }
}