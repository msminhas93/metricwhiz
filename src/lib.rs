use polars::prelude::*;
use std::collections::BTreeMap;
use std::error::Error;

pub struct BinaryClsEvaluator {
    pub pred_df: DataFrame,
    pub threshold: f64,
    pub metrics_cache: BTreeMap<i64, Metrics>,
}
impl BinaryClsEvaluator {
    /// Creates a new `BinaryClsEvaluator` from a CSV file.
    pub fn new(pred_file_path: &str) -> Result<Self, Box<dyn Error>> {
        let pred_df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(pred_file_path.into()))
            .unwrap()
            .finish()
            .unwrap();
        if !pred_df.get_column_names().contains(&"pred_score")
            || !pred_df.get_column_names().contains(&"ground_truth")
        {
            return Err(
                "The CSV file must contain 'pred_score' and 'ground_truth' columns.".into(),
            );
        }

        Ok(BinaryClsEvaluator {
            pred_df,
            threshold: 0.5,
            metrics_cache: BTreeMap::new(),
        })
    }

    /// Sets the predicted labels based on the current threshold.
    pub fn set_pred_label(&mut self) -> Result<(), PolarsError> {
        let threshold = self.threshold;
        self.pred_df = self
            .pred_df
            .clone()
            .lazy()
            .with_column(
                when(col("pred_score").gt(lit(threshold)))
                    .then(lit(1i64))
                    .otherwise(lit(0i64))
                    .alias("pred_label"),
            )
            .collect()?;

        Ok(())
    }

    pub fn set_threshold(&mut self, threshold: f64) -> Result<(), Box<dyn Error>> {
        self.threshold = (threshold * 100.0).round() / 100.0;
        self.set_pred_label()?;
        Ok(())
    }

    /// Calculates the confusion matrix.
    pub fn calculate_confusion_matrix(&self) -> Result<(usize, usize, usize, usize), PolarsError> {
        let confusion_matrix = self
            .pred_df
            .clone()
            .lazy()
            .with_columns([
                col("ground_truth")
                    .eq(lit(true))
                    .and(col("pred_label").eq(lit(true)))
                    .alias("tp"),
                col("ground_truth")
                    .eq(lit(false))
                    .and(col("pred_label").eq(lit(false)))
                    .alias("tn"),
                col("ground_truth")
                    .eq(lit(false))
                    .and(col("pred_label").eq(lit(true)))
                    .alias("fp"),
                col("ground_truth")
                    .eq(lit(true))
                    .and(col("pred_label").eq(lit(false)))
                    .alias("fn"),
            ])
            .select([
                col("tp").sum(),
                col("tn").sum(),
                col("fp").sum(),
                col("fn").sum(),
            ])
            .collect()?;

        let tp = confusion_matrix.column("tp")?.sum::<i64>().unwrap_or(0) as usize;
        let tn = confusion_matrix.column("tn")?.sum::<i64>().unwrap_or(0) as usize;
        let fp = confusion_matrix.column("fp")?.sum::<i64>().unwrap_or(0) as usize;
        let fn_ = confusion_matrix.column("fn")?.sum::<i64>().unwrap_or(0) as usize;

        Ok((tp, tn, fp, fn_))
    }
    /// Calculates precision, recall, and F1 score.
    pub fn calculate_precision_recall_f1(&self) -> Result<(f64, f64, f64), PolarsError> {
        let (tp, _, fp, fn_) = self.calculate_confusion_matrix()?;

        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / (tp + fn_) as f64;
        let f1_score = 2.0 * (precision * recall) / (precision + recall);

        Ok((precision, recall, f1_score))
    }

    /// Calculates accuracy.
    pub fn calculate_accuracy(&self) -> Result<f64, PolarsError> {
        let (tp, tn, fp, fn_) = self.calculate_confusion_matrix()?;
        let accuracy = (tp + tn) as f64 / (tp + tn + fp + fn_) as f64;
        Ok(accuracy)
    }

    /// Calculates specificity.
    pub fn calculate_specificity(&self) -> Result<f64, PolarsError> {
        let (_, tn, fp, _) = self.calculate_confusion_matrix()?;
        let specificity = tn as f64 / (tn + fp) as f64;
        Ok(specificity)
    }

    /// Calculates Matthews correlation coefficient (MCC).
    pub fn calculate_mcc(&self) -> Result<f64, PolarsError> {
        let (tp, tn, fp, fn_) = self.calculate_confusion_matrix()?;
        let numerator = (tp * tn) as f64 - (fp * fn_) as f64;
        let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)) as f64;
        let mcc = if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator.sqrt()
        };
        Ok(mcc)
    }
    pub fn calculate_auroc(&self) -> Result<f64, Box<dyn Error>> {
        // Extract ground truth and prediction scores
        let ground_truth = self.pred_df.column("ground_truth")?.i64()?;
        let pred_scores = self.pred_df.column("pred_score")?.f64()?;

        // Collect data into vectors
        let mut data: Vec<(f64, bool)> = ground_truth
            .into_iter()
            .zip(pred_scores.into_iter())
            .filter_map(|(gt, score)| {
                if let (Some(gt), Some(score)) = (gt, score) {
                    Some((score, gt == 1)) // Convert i64 to bool: 1 is true, everything else is false
                } else {
                    None
                }
            })
            .collect();

        // Sort by prediction score in descending order
        data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut auc = 0.0;
        let mut true_positives = 0;
        let mut false_positives = 0;
        let total_positives = data.iter().filter(|&&(_, gt)| gt).count();
        let total_negatives = data.len() - total_positives;

        if total_positives == 0 || total_negatives == 0 {
            return Ok(0.5); // AUROC is undefined, return 0.5
        }

        let mut prev_tpr = 0.0;
        let mut prev_fpr = 0.0;

        for (_, is_positive) in data {
            if is_positive {
                true_positives += 1;
            } else {
                false_positives += 1;
            }

            let tpr = true_positives as f64 / total_positives as f64;
            let fpr = false_positives as f64 / total_negatives as f64;

            // Calculate trapezoid area
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

            prev_tpr = tpr;
            prev_fpr = fpr;
        }

        Ok(auc)
    }
    pub fn classification_report(&self) -> Result<String, Box<dyn Error>> {
        let (tp, tn, fp, fn_) = self.calculate_confusion_matrix()?;
        let (precision, recall, f1_score) = self.calculate_precision_recall_f1()?;
        let accuracy = self.calculate_accuracy()?;

        let support_0 = tn + fp;
        let support_1 = tp + fn_;
        let total_support = support_0 + support_1;

        let precision_0 = tn as f64 / (tn + fn_) as f64;
        let recall_0 = tn as f64 / (tn + fp) as f64;
        let f1_score_0 = 2.0 * (precision_0 * recall_0) / (precision_0 + recall_0);

        let macro_precision = (precision + precision_0) / 2.0;
        let macro_recall = (recall + recall_0) / 2.0;
        let macro_f1 = (f1_score + f1_score_0) / 2.0;

        let weighted_precision =
            (precision * support_1 as f64 + precision_0 * support_0 as f64) / total_support as f64;
        let weighted_recall =
            (recall * support_1 as f64 + recall_0 * support_0 as f64) / total_support as f64;
        let weighted_f1 =
            (f1_score * support_1 as f64 + f1_score_0 * support_0 as f64) / total_support as f64;

        let report = format!(
            "⠀⠀⠀          precision   recall  f1-score   support\n\n\
             0             {:.3}      {:.3}     {:.3}       {}\n\
             1             {:.3}      {:.3}     {:.3}       {}\n\n\
             accuracy                           {:.3}       {}\n\
             macro avg     {:.3}      {:.3}     {:.3}       {}\n\
             weighted avg  {:.3}      {:.3}     {:.3}       {}",
            precision_0,
            recall_0,
            f1_score_0,
            support_0,
            precision,
            recall,
            f1_score,
            support_1,
            accuracy,
            total_support,
            macro_precision,
            macro_recall,
            macro_f1,
            total_support,
            weighted_precision,
            weighted_recall,
            weighted_f1,
            total_support
        );

        Ok(report)
    }
    pub fn calculate_metrics(&mut self) -> Result<Metrics, Box<dyn Error>> {
        let threshold_key = (self.threshold * 100.0).round() as i64;

        if !self.metrics_cache.contains_key(&threshold_key) {
            let (tp, tn, fp, fn_) = self.calculate_confusion_matrix()?;
            let (precision, recall, f1_score) = self.calculate_precision_recall_f1()?;
            let accuracy = self.calculate_accuracy()?;
            let specificity = self.calculate_specificity()?;
            let mcc = self.calculate_mcc()?;
            let auroc = self.calculate_auroc()?;
            let classification_report = self.classification_report()?;

            let metrics = Metrics {
                tp,
                tn,
                fp,
                fn_,
                precision,
                recall,
                f1_score,
                accuracy,
                specificity,
                mcc,
                auroc,
                classification_report,
            };

            self.metrics_cache.insert(threshold_key, metrics);
        }

        Ok(self.metrics_cache.get(&threshold_key).unwrap().clone())
    }

    pub fn calculate_optimal_thresholds(&mut self) -> Result<OptimalThresholds, Box<dyn Error>> {
        let thresholds: Vec<f64> = (1..99).map(|i| i as f64 / 100.0).collect();
        let original_threshold = self.threshold;

        let mut optimal = OptimalThresholds {
            precision: OptimalMetric {
                threshold: 0.0,
                value: f64::NEG_INFINITY,
            },
            recall: OptimalMetric {
                threshold: 0.0,
                value: f64::NEG_INFINITY,
            },
            f1_score: OptimalMetric {
                threshold: 0.0,
                value: f64::NEG_INFINITY,
            },
            accuracy: OptimalMetric {
                threshold: 0.0,
                value: f64::NEG_INFINITY,
            },
            specificity: OptimalMetric {
                threshold: 0.0,
                value: f64::NEG_INFINITY,
            },
            mcc: OptimalMetric {
                threshold: 0.0,
                value: f64::NEG_INFINITY,
            },
        };

        for &threshold in &thresholds {
            self.set_threshold(threshold)?;
            let metrics = self.calculate_metrics()?;

            if metrics.precision > optimal.precision.value {
                optimal.precision = OptimalMetric {
                    threshold,
                    value: metrics.precision,
                };
            }
            if metrics.recall > optimal.recall.value {
                optimal.recall = OptimalMetric {
                    threshold,
                    value: metrics.recall,
                };
            }
            if metrics.f1_score > optimal.f1_score.value {
                optimal.f1_score = OptimalMetric {
                    threshold,
                    value: metrics.f1_score,
                };
            }
            if metrics.accuracy > optimal.accuracy.value {
                optimal.accuracy = OptimalMetric {
                    threshold,
                    value: metrics.accuracy,
                };
            }
            if metrics.specificity > optimal.specificity.value {
                optimal.specificity = OptimalMetric {
                    threshold,
                    value: metrics.specificity,
                };
            }
            if metrics.mcc > optimal.mcc.value {
                optimal.mcc = OptimalMetric {
                    threshold,
                    value: metrics.mcc,
                };
            }
        }

        // Restore the original threshold
        self.set_threshold(original_threshold)?;

        Ok(optimal)
    }
}

#[derive(Debug, Default)]
pub struct OptimalThresholds {
    pub precision: OptimalMetric,
    pub recall: OptimalMetric,
    pub f1_score: OptimalMetric,
    pub accuracy: OptimalMetric,
    pub specificity: OptimalMetric,
    pub mcc: OptimalMetric,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OptimalMetric {
    pub threshold: f64,
    pub value: f64,
}

// Struct to hold all metrics
#[derive(Clone)]
pub struct Metrics {
    pub tp: usize,
    pub tn: usize,
    pub fp: usize,
    pub fn_: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy: f64,
    pub specificity: f64,
    pub mcc: f64,
    pub auroc: f64,
    pub classification_report: String,
}
