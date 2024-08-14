use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use polars::prelude::*;
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table},
    Terminal,
};
use std::error::Error;
use std::io::stdout;

use std::collections::BTreeMap;

pub struct BinaryClsEvaluator {
    pred_df: DataFrame,
    threshold: f64,
    metrics_cache: BTreeMap<i64, Metrics>,
}
impl BinaryClsEvaluator {
    /// Creates a new `BinaryClsEvaluator` from a CSV file.
    fn new(pred_file_path: &str) -> Result<Self, Box<dyn Error>> {
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
    fn set_pred_label(&mut self) -> Result<(), PolarsError> {
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
    fn calculate_confusion_matrix(&self) -> Result<(usize, usize, usize, usize), PolarsError> {
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
    fn calculate_precision_recall_f1(&self) -> Result<(f64, f64, f64), PolarsError> {
        let (tp, _, fp, fn_) = self.calculate_confusion_matrix()?;

        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / (tp + fn_) as f64;
        let f1_score = 2.0 * (precision * recall) / (precision + recall);

        Ok((precision, recall, f1_score))
    }

    /// Calculates accuracy.
    fn calculate_accuracy(&self) -> Result<f64, PolarsError> {
        let (tp, tn, fp, fn_) = self.calculate_confusion_matrix()?;
        let accuracy = (tp + tn) as f64 / (tp + tn + fp + fn_) as f64;
        Ok(accuracy)
    }

    /// Calculates specificity.
    fn calculate_specificity(&self) -> Result<f64, PolarsError> {
        let (_, tn, fp, _) = self.calculate_confusion_matrix()?;
        let specificity = tn as f64 / (tn + fp) as f64;
        Ok(specificity)
    }

    /// Calculates Matthews correlation coefficient (MCC).
    fn calculate_mcc(&self) -> Result<f64, PolarsError> {
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
    fn calculate_auroc(&self) -> Result<f64, Box<dyn Error>> {
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
    fn classification_report(&self) -> Result<String, Box<dyn Error>> {
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
            "              precision    recall  f1-score   support\n\n\
             0             {:.2}      {:.2}     {:.2}        {}\n\
             1             {:.2}      {:.2}     {:.2}        {}\n\n\
             accuracy                         {:.2}        {}\n\
             macro avg     {:.2}      {:.2}     {:.2}        {}\n\
             weighted avg  {:.2}      {:.2}     {:.2}        {}",
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
            precision: (0.0, f64::NEG_INFINITY),
            recall: (0.0, f64::NEG_INFINITY),
            f1_score: (0.0, f64::NEG_INFINITY),
            accuracy: (0.0, f64::NEG_INFINITY),
            specificity: (0.0, f64::NEG_INFINITY),
            mcc: (0.0, f64::NEG_INFINITY),
        };

        for &threshold in &thresholds {
            self.set_threshold(threshold)?;
            let metrics = self.calculate_metrics()?;

            // Update optimal thresholds...
            if metrics.precision > optimal.precision.1 {
                optimal.precision = (threshold, metrics.precision);
            }
            if metrics.recall > optimal.recall.1 {
                optimal.recall = (threshold, metrics.recall);
            }
            if metrics.f1_score > optimal.f1_score.1 {
                optimal.f1_score = (threshold, metrics.f1_score);
            }
            if metrics.accuracy > optimal.accuracy.1 {
                optimal.accuracy = (threshold, metrics.accuracy);
            }
            if metrics.specificity > optimal.specificity.1 {
                optimal.specificity = (threshold, metrics.specificity);
            }
            if metrics.mcc > optimal.mcc.1 {
                optimal.mcc = (threshold, metrics.mcc);
            }
        }

        // Restore the original threshold
        self.set_threshold(original_threshold)?;

        Ok(optimal)
    }
}

#[derive(Default)]
pub struct OptimalThresholds {
    precision: (f64, f64),
    recall: (f64, f64),
    f1_score: (f64, f64),
    accuracy: (f64, f64),
    specificity: (f64, f64),
    mcc: (f64, f64),
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

pub fn draw_ui<B: Backend>(
    terminal: &mut Terminal<B>,
    evaluator: &mut BinaryClsEvaluator,
) -> Result<(), Box<dyn Error>> {
    let mut optimal_thresholds = evaluator.calculate_optimal_thresholds()?;

    loop {
        let threshold = evaluator.threshold;
        let metrics = evaluator.calculate_metrics()?;

        terminal.draw(|f| {
            let size = f.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints(
                    [
                        Constraint::Length(3),
                        Constraint::Percentage(15),
                        Constraint::Percentage(20),
                        Constraint::Percentage(20),
                        Constraint::Percentage(45),
                    ]
                    .as_ref(),
                )
                .split(size);

            // Threshold selector
            let gauge = Gauge::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Threshold Selector"),
                )
                .gauge_style(Style::default().fg(Color::Yellow))
                .ratio(threshold)
                .label(format!("{:.2}", threshold));
            f.render_widget(gauge, chunks[0]);

            // Confusion Matrix
            let confusion_matrix = Table::new(
                vec![
                    Row::new(vec![
                        Cell::from(""),
                        Cell::from("Predicted 0"),
                        Cell::from("Predicted 1"),
                    ]),
                    Row::new(vec![
                        Cell::from("Actual 0"),
                        Cell::from(format!("{}", metrics.tn)),
                        Cell::from(format!("{}", metrics.fp)),
                    ]),
                    Row::new(vec![
                        Cell::from("Actual 1"),
                        Cell::from(format!("{}", metrics.fn_)),
                        Cell::from(format!("{}", metrics.tp)),
                    ]),
                ],
                [
                    Constraint::Percentage(33),
                    Constraint::Percentage(33),
                    Constraint::Percentage(34),
                ]
                .as_ref(),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Confusion Matrix"),
            );
            f.render_widget(confusion_matrix, chunks[1]);

            // Metrics
            let metrics_text = vec![
                Line::from(Span::styled(
                    format!("Precision: {:.4}", metrics.precision),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!("Recall: {:.4}", metrics.recall),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!("F1 Score: {:.4}", metrics.f1_score),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!("Accuracy: {:.4}", metrics.accuracy),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!("Specificity: {:.4}", metrics.specificity),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!("MCC: {:.4}", metrics.mcc),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!("AUROC: {:.4}", metrics.auroc),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
            ];
            let metrics_paragraph = Paragraph::new(Text::from(metrics_text))
                .block(Block::default().borders(Borders::ALL).title("Metrics"))
                .wrap(ratatui::widgets::Wrap { trim: true });
            f.render_widget(metrics_paragraph, chunks[2]);

            // Optimal Thresholds
            let optimal_text = vec![
                Line::from(Span::styled(
                    format!(
                        "Precision: {:.3} (value: {:.4})",
                        optimal_thresholds.precision.0, optimal_thresholds.precision.1
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Recall: {:.3} (value: {:.4})",
                        optimal_thresholds.recall.0, optimal_thresholds.recall.1
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "F1 Score: {:.3} (value: {:.4})",
                        optimal_thresholds.f1_score.0, optimal_thresholds.f1_score.1
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Accuracy: {:.3} (value: {:.4})",
                        optimal_thresholds.accuracy.0, optimal_thresholds.accuracy.1
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Specificity: {:.3} (value: {:.4})",
                        optimal_thresholds.specificity.0, optimal_thresholds.specificity.1
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "MCC: {:.3} (value: {:.4})",
                        optimal_thresholds.mcc.0, optimal_thresholds.mcc.1
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
            ];
            let optimal_paragraph = Paragraph::new(Text::from(optimal_text))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Optimal Thresholds"),
                )
                .wrap(ratatui::widgets::Wrap { trim: true });
            f.render_widget(optimal_paragraph, chunks[3]);

            // Classification Report
            let report_paragraph =
                Paragraph::new(Text::from(metrics.classification_report.clone()))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Classification Report"),
                    )
                    .wrap(ratatui::widgets::Wrap { trim: true });
            f.render_widget(report_paragraph, chunks[4]);
        })?;

        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') => break,
                KeyCode::Left => {
                    let new_threshold = (threshold - 0.01).max(0.0);
                    if new_threshold != threshold {
                        evaluator.set_threshold(new_threshold)?;
                    }
                }
                KeyCode::Right => {
                    let new_threshold = (threshold + 0.01).min(1.0);
                    if new_threshold != threshold {
                        evaluator.set_threshold(new_threshold)?;
                    }
                }
                KeyCode::Char('r') => {
                    // Recalculate optimal thresholds
                    optimal_thresholds = evaluator.calculate_optimal_thresholds()?;
                }
                _ => {}
            }
        }
    }

    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut evaluator =
        BinaryClsEvaluator::new(r#"C:\Users\msmin\code\perf_eval\tests\sample_pred_file.csv"#)?;
    evaluator.set_threshold(0.5)?;

    terminal::enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Draw the UI
    draw_ui(&mut terminal, &mut evaluator)?;

    // Restore the terminal
    terminal::disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        // DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}
