use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use polars::prelude::*;

use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Terminal,
};
use std::error::Error;
/// A struct for evaluating binary classification models.
struct BinaryClsEvaluator {
    pred_df: DataFrame,
    threshold: f64,
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
        // Print the first 5 rows of the DataFrame
        println!("DataFrame head:\n{}", pred_df.head(Some(5)));
        Ok(BinaryClsEvaluator {
            pred_df,
            threshold: 0.5,
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

        // Print the first 5 rows of the updated DataFrame
        println!("Updated DataFrame head:\n{}", self.pred_df.head(Some(5)));

        Ok(())
    }

    /// Calculates the confusion matrix.
    fn calculate_confusion_matrix(&self) -> Result<(usize, usize, usize, usize), PolarsError> {
        let y_true = self.pred_df.column("ground_truth")?.i32()?;
        let y_pred = self.pred_df.column("pred_label")?.i32()?;

        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_ = 0;

        for (true_label, pred_label) in y_true.into_iter().zip(y_pred.into_iter()) {
            match (true_label, pred_label) {
                (Some(1), Some(1)) => tp += 1,
                (Some(0), Some(0)) => tn += 1,
                (Some(0), Some(1)) => fp += 1,
                (Some(1), Some(0)) => fn_ += 1,
                _ => {}
            }
        }

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
}

fn draw_ui<B: Backend>(
    terminal: &mut Terminal<B>,
    evaluator: &BinaryClsEvaluator,
) -> Result<(), Box<dyn Error>> {
    let (tp, tn, fp, fn_) = evaluator.calculate_confusion_matrix()?;
    let (precision, recall, f1_score) = evaluator.calculate_precision_recall_f1()?;
    let accuracy = evaluator.calculate_accuracy()?;
    let specificity = evaluator.calculate_specificity()?;
    let mcc = evaluator.calculate_mcc()?;

    terminal.draw(|f| {
        let size = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Percentage(30), Constraint::Percentage(70)].as_ref())
            .split(size);

        let confusion_matrix = Table::new(
            vec![
                Row::new(vec![
                    Cell::from(""),
                    Cell::from("Predicted 0"),
                    Cell::from("Predicted 1"),
                ]),
                Row::new(vec![
                    Cell::from("Actual 0"),
                    Cell::from(format!("{}", tn)),
                    Cell::from(format!("{}", fp)),
                ]),
                Row::new(vec![
                    Cell::from("Actual 1"),
                    Cell::from(format!("{}", fn_)),
                    Cell::from(format!("{}", tp)),
                ]),
            ],
            [
                Constraint::Percentage(33),
                Constraint::Percentage(33),
                Constraint::Percentage(33),
            ]
            .as_ref(),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Confusion Matrix"),
        );

        f.render_widget(confusion_matrix, chunks[0]);

        let metrics = Text::from(vec![
            Line::from(Span::styled(
                format!("Precision: {:.2}", precision),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("Recall: {:.2}", recall),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("F1 Score: {:.2}", f1_score),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("Accuracy: {:.2}", accuracy),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("Specificity: {:.2}", specificity),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("MCC: {:.2}", mcc),
                Style::default().add_modifier(Modifier::BOLD),
            )),
        ]);

        let metrics_paragraph = Paragraph::new(metrics)
            .block(Block::default().borders(Borders::ALL).title("Metrics"))
            .wrap(ratatui::widgets::Wrap { trim: true });

        f.render_widget(metrics_paragraph, chunks[1]);
    })?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut evaluator =
        BinaryClsEvaluator::new(r#"C:\Users\msmin\code\perf_eval\tests\sample_pred_file.csv"#)?;
    evaluator.set_pred_label()?;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Draw the UI
    draw_ui(&mut terminal, &evaluator)?;

    // Wait for a key press to exit
    loop {
        if let Event::Key(key) = event::read()? {
            if key.code == KeyCode::Char('q') {
                break;
            }
        }
    }

    // Restore the terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableMouseCapture)?;
    terminal.show_cursor()?;

    Ok(())
}
