use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use metricwhiz::BinaryClsEvaluator;
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
                        "Optimal Precision: {:.4} (Threshold: {:.4})",
                        optimal_thresholds.precision.value, optimal_thresholds.precision.threshold
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Optimal Recall: {:.4} (Threshold: {:.4})",
                        optimal_thresholds.recall.value, optimal_thresholds.recall.threshold
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Optimal F1 Score: {:.4} (Threshold: {:.4})",
                        optimal_thresholds.f1_score.value, optimal_thresholds.f1_score.threshold
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Optimal Accuracy: {:.4} (Threshold: {:.4})",
                        optimal_thresholds.accuracy.value, optimal_thresholds.accuracy.threshold
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Optimal Specificity: {:.4} (Threshold: {:.4})",
                        optimal_thresholds.specificity.value,
                        optimal_thresholds.specificity.threshold
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled(
                    format!(
                        "Optimal MCC: {:.4} (Threshold: {:.4})",
                        optimal_thresholds.mcc.value, optimal_thresholds.mcc.threshold
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
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}
