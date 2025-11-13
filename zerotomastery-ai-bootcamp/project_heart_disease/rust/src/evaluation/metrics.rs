use ndarray::Array1;

pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
  if y_true.len() != y_pred.len() {
    panic!("Arrays must have the same length");
  }

  let correct = y_true
    .iter()
    .zip(y_pred.iter())
    .filter(|(true_val, pred_val)| true_val == pred_val)
    .count();

  correct as f64 / y_true.len() as f64
}

pub fn confusion_matrix(
  y_true: &Array1<f64>,
  y_pred: &Array1<f64>,
) -> (usize, usize, usize, usize) {
  if y_true.len() != y_pred.len() {
    panic!("Arrays must have the same length");
  }

  let mut true_negatives = 0;
  let mut false_positives = 0;
  let mut false_negatives = 0;
  let mut true_positives = 0;

  for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
    match (*true_val, *pred_val) {
      (0.0, 0.0) => true_negatives += 1,
      (0.0, 1.0) => false_positives += 1,
      (1.0, 0.0) => false_negatives += 1,
      (1.0, 1.0) => true_positives += 1,
      _ => panic!("Values must be 0.0 or 1.0"),
    }
  }

  (
    true_negatives,
    false_positives,
    false_negatives,
    true_positives,
  )
}

pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
  let (_tn, fp, _fn_, tp) = confusion_matrix(y_true, y_pred);

  if tp + fp == 0 {
    0.0
  } else {
    tp as f64 / (tp + fp) as f64
  }
}

pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
  let (_tn, _fp, fn_, tp) = confusion_matrix(y_true, y_pred);

  if tp + fn_ == 0 {
    0.0
  } else {
    tp as f64 / (tp + fn_) as f64
  }
}

pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
  let prec = precision(y_true, y_pred);
  let rec = recall(y_true, y_pred);

  if prec + rec == 0.0 {
    0.0
  } else {
    2.0 * (prec * rec) / (prec + rec)
  }
}
