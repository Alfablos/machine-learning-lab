use anyhow::Result;
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct HeartDiseaseRecord {
  age: f64,
  sex: f64,
  cp: f64,
  trestbps: f64,
  chol: f64,
  fbs: f64,
  restecg: f64,
  thalach: f64,
  exang: f64,
  oldpeak: f64,
  slope: f64,
  ca: f64,
  thal: f64,
  target: f64,
}

pub struct Dataset {
  pub features: Array2<f64>,
  pub targets: Array1<f64>,
  pub feature_names: Vec<String>,
}

impl Dataset {
  pub fn new(features: Array2<f64>, targets: Array1<f64>) -> Self {
    let feature_names = vec![
      "age".to_string(),
      "sex".to_string(),
      "cp".to_string(),
      "trestbps".to_string(),
      "chol".to_string(),
      "fbs".to_string(),
      "restecg".to_string(),
      "thalach".to_string(),
      "exang".to_string(),
      "oldpeak".to_string(),
      "slope".to_string(),
      "ca".to_string(),
      "thal".to_string(),
    ];

    Self {
      features,
      targets,
      feature_names,
    }
  }

  pub fn n_samples(&self) -> usize {
    self.features.nrows()
  }

  pub fn n_features(&self) -> usize {
    self.features.ncols()
  }
}

pub fn load_heart_disease_data<P: AsRef<Path>>(path: P) -> Result<Dataset> {
  let file = File::open(path)?;
  let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

  let mut records: Vec<HeartDiseaseRecord> = Vec::new();

  for result in reader.deserialize() {
    let record: HeartDiseaseRecord = result?;
    records.push(record);
  }

  let n_samples = records.len();
  let n_features = 13;

  let mut features = Array2::zeros((n_samples, n_features));
  let mut targets = Array1::zeros(n_samples);

  for (i, record) in records.iter().enumerate() {
    features[(i, 0)] = record.age;
    features[(i, 1)] = record.sex;
    features[(i, 2)] = record.cp;
    features[(i, 3)] = record.trestbps;
    features[(i, 4)] = record.chol;
    features[(i, 5)] = record.fbs;
    features[(i, 6)] = record.restecg;
    features[(i, 7)] = record.thalach;
    features[(i, 8)] = record.exang;
    features[(i, 9)] = record.oldpeak;
    features[(i, 10)] = record.slope;
    features[(i, 11)] = record.ca;
    features[(i, 12)] = record.thal;

    targets[i] = record.target;
  }

  Ok(Dataset::new(features, targets))
}
