mod data;
mod evaluation;
mod models;

use anyhow::Result;
use data::{load_heart_disease_data, train_test_split, StandardScaler};
use evaluation::{accuracy, f1_score, precision, recall};
use models::{LogisticRegressionModel, NonTrainedModel, RandomForestModel};



fn main() -> Result<()> {
  println!("Heart Disease Classifier - Rust Implementation");
  println!("============================================");

  // Load data
  println!("Loading dataset...");
  let dataset = load_heart_disease_data("data/heart-disease.csv")?;
  println!(
    "Dataset loaded: {} samples, {} features",
    dataset.n_samples(),
    dataset.n_features()
  );

  // Split data
  println!("Splitting data into train/test sets...");
  let split = train_test_split(&dataset.features, &dataset.targets, 0.2)?;
  println!("Train set: {} samples", split.x_train.nrows());
  println!("Test set: {} samples", split.x_test.nrows());

  // Scale features
  println!("Scaling features...");
  let mut scaler = StandardScaler::new();
  let x_train_scaled = scaler.fit_transform(&split.x_train);
  let x_test_scaled = scaler.transform(&split.x_test);

  // Train and evaluate Logistic Regression
  println!("\nTraining Logistic Regression...");
  let lr_model = LogisticRegressionModel::new();
  let lr_fitted = lr_model.train(&x_train_scaled, &split.y_train)?;
  let lr_predictions = lr_fitted.predict(&x_test_scaled)?;

  let lr_accuracy = accuracy(&split.y_test, &lr_predictions);
  let lr_precision = precision(&split.y_test, &lr_predictions);
  let lr_recall = recall(&split.y_test, &lr_predictions);
  let lr_f1 = f1_score(&split.y_test, &lr_predictions);

  println!("Logistic Regression Results:");
  println!("  Accuracy: {:.3}", lr_accuracy);
  println!("  Precision: {:.3}", lr_precision);
  println!("  Recall: {:.3}", lr_recall);
  println!("  F1 Score: {:.3}", lr_f1);

  // Train and evaluate Random Forest
  println!("\nTraining Random Forest...");
  let rf_model = RandomForestModel::new(Some(100), Some(10), Some(2), Some(1), None);
  let rf_fitted = rf_model.train(&x_train_scaled, &split.y_train)?;
  let rf_predictions = rf_fitted.predict(&x_test_scaled)?;

  let rf_accuracy = accuracy(&split.y_test, &rf_predictions);
  let rf_precision = precision(&split.y_test, &rf_predictions);
  let rf_recall = recall(&split.y_test, &rf_predictions);
  let rf_f1 = f1_score(&split.y_test, &rf_predictions);

  println!("Random Forest Results:");
  println!("  Accuracy: {:.3}", rf_accuracy);
  println!("  Precision: {:.3}", rf_precision);
  println!("  Recall: {:.3}", rf_recall);
  println!("  F1 Score: {:.3}", rf_f1);

  // Summary
  println!("\n=== SUMMARY ===");
  println!("Target accuracies from Python implementation:");
  println!("  Logistic Regression: 85.2%");
  println!("  Random Forest: 88.5%");
  println!("\nRust implementation results:");
  println!("  Logistic Regression: {:.1}%", lr_accuracy * 100.0);
  println!("  Random Forest: {:.1}%", rf_accuracy * 100.0);

  Ok(())
}
