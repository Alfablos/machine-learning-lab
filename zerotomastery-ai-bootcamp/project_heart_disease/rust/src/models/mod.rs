pub mod logistic_regression;
pub mod random_forest;

use anyhow::Result;

pub use logistic_regression::*;
use ndarray::{Array1, Array2};
pub use random_forest::*;

pub trait TrainStatus {}

pub struct Trained;
impl TrainStatus for Trained {}

pub struct NonTrained;
impl TrainStatus for NonTrained {}

// Main trait with GATs
pub trait NonTrainedModel {
  fn train(self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<Box<dyn TrainedModel>>;
}

pub trait TrainedModel {
  // Only implementable when State = Trained
  fn predict(&self, x_test: &Array2<f64>) -> Result<Array1<f64>>;

  fn predict_proba(&self, x_test: &Array2<f64>) -> Result<Array2<f64>>;
}

pub enum SupportedClassifiers {
  LogisticRegression {
    alpha: Option<f64>,
    max_iter: Option<usize>,
  },
  RandomForestClassifier {
    n_trees: Option<usize>,
    max_depth: Option<usize>,
    min_samples_split: Option<usize>,
    min_samples_leaf: Option<usize>,
    max_features: Option<usize>,
  }
}

pub fn new_classification_model(clf: SupportedClassifiers) -> Box<dyn NonTrainedModel> {
  match clf {
    SupportedClassifiers::LogisticRegression { alpha, max_iter } => {
      let mut model = LogisticRegressionModel::new();
      if let Some(alpha) = alpha {
        model = model.with_alpha(alpha)
      }
      if let Some(max_iter) = max_iter {
        model = model.with_max_iter(max_iter)
      }
      Box::new(model)
    }
    SupportedClassifiers::RandomForestClassifier { n_trees, max_depth, min_samples_split, min_samples_leaf, max_features } => {
      Box::new(RandomForestModel::new(n_trees, max_depth, min_samples_split, min_samples_leaf, max_features))
    }
  }
}

/*
* ### 1. The Rust Book

• Chapter 10: Generic Types, Traits, and Lifetimes
 • Traits: Defining Shared Behavior https://doc.rust-lang.org/book/ch10-02-traits.html
 • Using Trait Bounds to Conditionally Implement Methods https://doc.rust-lang.org/book/ch10-02-traits.html#using-trait-bounds-to-conditionally-implement-methods
• Chapter 19: Advanced Features
 • Advanced Lifetimes https://doc.rust-lang.org/book/ch19-02-advanced-lifetimes.html
 • Advanced Traits https://doc.rust-lang.org/book/ch19-03-advanced-traits.html


### 2. Rust by Example

• Generics https://doc.rust-lang.org/rust-by-example/generics.html
• Traits https://doc.rust-lang.org/rust-by-example/trait.html
• Advanced Traits https://doc.rust-lang.org/rust-by-example/trait/advanced.html

## Generic Associated Types (GATs) Documentation:

### 3. Rust RFC for GATs

• RFC 1598: Generic Associated Types https://rust-lang.github.io/rfcs/1598-generic_associated_types.html
 • The original proposal that introduced GATs to Rust


### 4. Rust Reference

• Generic Associated Types https://doc.rust-lang.org/reference/items/associated-items.html#generic-associated-types
 • Official language reference documentation


## Type-State Pattern Resources:

### 5. Blog Posts & Articles

• "The Type State Pattern in Rust" - Various blog posts on Medium and dev.to
• "Phantom Types in Rust" - Search for articles about phantom type parameters
• "Type-level Programming in Rust" - Advanced pattern articles

### 6. Conference Talks

• RustConf talks on advanced trait patterns
• RustFest presentations on type-level programming
• YouTube: Search for "Rust type state pattern" and "Rust GATs"


## Practical Examples:

### 7. GitHub Repositories

• Rust Cookbook: Look for advanced trait examples
• Rust-lang GitHub: Search for examples using GATs
• Open source Rust projects: Many use similar patterns for state machines

### 8. Stack Overflow

• Search for: "rust generic associated types", "rust type state pattern", "rust phantom types"
• Many practical Q&A about implementing these patterns

## Key Search Terms:

• "Rust Generic Associated Types"
• "Rust Type State Pattern"
• "Rust Phantom Types"
• "Rust Compile-time State Machine"
• "Rust Advanced Trait Patterns"

## Learning Path Recommendation:

1. Start with Rust Book chapters on traits and generics
2. Read the GATs RFC to understand the motivation
3. Look at practical examples in blog posts
4. Study open source code using similar patterns
5. Practice with small examples before applying to larger projects

*/
