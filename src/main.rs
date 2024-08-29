use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use metricwhiz::BinaryClsEvaluator;
use polars::prelude::*;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Cell, Gauge, List, ListItem, ListState, Paragraph, Row, Table, Tabs,
    },
    Terminal,
};
use std::error::Error;
use std::io::stdout;
use strum::{EnumIter, FromRepr, IntoEnumIterator};

fn main() -> Result<(), Box<dyn Error>> {
    terminal::enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut evaluator =
        BinaryClsEvaluator::new(r#"C:\Users\msmin\code\perf_eval\tests\sample_pred_file.csv"#)?;
    evaluator.set_threshold(0.5)?;

    let app_result = App::new(evaluator).run(&mut terminal);

    terminal::disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    app_result
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppState {
    Running,
    Quitting,
}

#[derive(Clone)]
struct Sample {
    id: usize, // or whatever type your ID is
    ground_truth: i64,
    pred_label: i64,
    details: String, // Add any other fields you need
}

#[derive(Clone, Copy, EnumIter, FromRepr, Debug, PartialEq)]
enum SelectedTab {
    ReportViewer,
    SampleViewer,
}

struct App {
    state: AppState,
    selected_tab: SelectedTab,
    evaluator: BinaryClsEvaluator,
    sample_list_state: ListState,
}

impl App {
    fn new(evaluator: BinaryClsEvaluator) -> Self {
        App {
            state: AppState::Running,
            selected_tab: SelectedTab::ReportViewer,
            evaluator,
            sample_list_state: ListState::default(),
        }
    }

    fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> Result<(), Box<dyn Error>> {
        while self.state == AppState::Running {
            terminal.draw(|f| self.draw_ui(f))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn handle_events(&mut self) -> std::io::Result<()> {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => self.quit(),
                KeyCode::Char('p') => self.previous_tab(),
                KeyCode::Char('n') => self.next_tab(),
                KeyCode::Left | KeyCode::Right => {
                    if self.selected_tab == SelectedTab::ReportViewer {
                        self.handle_threshold_events(key.code);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn handle_threshold_events(&mut self, key: KeyCode) {
        let threshold = self.evaluator.threshold;
        match key {
            KeyCode::Left => {
                let new_threshold = (threshold - 0.01).max(0.0);
                if new_threshold != threshold {
                    self.evaluator.set_threshold(new_threshold).unwrap();
                }
            }
            KeyCode::Right => {
                let new_threshold = (threshold + 0.01).min(1.0);
                if new_threshold != threshold {
                    self.evaluator.set_threshold(new_threshold).unwrap();
                }
            }
            _ => {}
        }
    }

    fn next_tab(&mut self) {
        self.selected_tab = self.selected_tab.next();
    }

    fn previous_tab(&mut self) {
        self.selected_tab = self.selected_tab.previous();
    }

    fn quit(&mut self) {
        self.state = AppState::Quitting;
    }

    fn draw_ui(&mut self, f: &mut ratatui::Frame) {
        let size = f.area();
        let chunks = Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
            .split(size);

        let titles = SelectedTab::iter().map(|tab| {
            Line::from(Span::styled(
                format!("{:?}", tab),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ))
        });
        let tabs = Tabs::new(titles)
            .select(self.selected_tab as usize)
            .block(Block::default().borders(Borders::ALL).title("Tabs"));
        f.render_widget(tabs, chunks[0]);

        match self.selected_tab {
            SelectedTab::ReportViewer => self.render_tab_0(f, chunks[1]),
            SelectedTab::SampleViewer => self.render_tab_1(f, chunks[1]),
        }
    }

    fn render_tab_0(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
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
            .split(area);

        let threshold = self.evaluator.threshold;
        let metrics = self.evaluator.calculate_metrics().unwrap();
        let optimal_thresholds = self.evaluator.calculate_optimal_thresholds().unwrap();

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
                    optimal_thresholds.specificity.value, optimal_thresholds.specificity.threshold
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
        let report_paragraph = Paragraph::new(Text::from(metrics.classification_report.clone()))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Classification Report"),
            )
            .wrap(ratatui::widgets::Wrap { trim: true });
        f.render_widget(report_paragraph, chunks[4]);
    }

    fn render_tab_1(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([Constraint::Percentage(30), Constraint::Percentage(70)].as_ref())
            .split(area);

        // Get filtered samples based on the selected category (tn, tp, fp, fn)
        let samples = self
            .get_filtered_samples("tp")
            .unwrap_or_else(|_| Vec::new());

        // Create a list of sample identifiers for the left pane
        let sample_list: Vec<ListItem> = samples
            .iter()
            .map(|sample| ListItem::new(format!("Sample ID: {}", sample.id)))
            .collect();

        // Render the list of samples
        let sample_list_widget = List::new(sample_list)
            .block(Block::default().borders(Borders::ALL).title("Samples"))
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">>");
        f.render_stateful_widget(sample_list_widget, chunks[0], &mut self.sample_list_state);

        // Render the details of the selected sample in the right pane
        if let Some(selected_sample) = self.get_selected_sample() {
            let sample_details = format!("Details:\n{}", selected_sample.details);
            let sample_paragraph = Paragraph::new(sample_details)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Sample Details"),
                )
                .wrap(ratatui::widgets::Wrap { trim: true });
            f.render_widget(sample_paragraph, chunks[1]);
        }
    }

    fn get_filtered_samples(&self, category: &str) -> Result<Vec<Sample>, PolarsError> {
        let lazy_df = self.evaluator.pred_df.clone().lazy();

        let filter_condition = match category {
            "tp" => col("ground_truth")
                .eq(lit(1))
                .and(col("pred_label").eq(lit(1))),
            "fp" => col("ground_truth")
                .eq(lit(0))
                .and(col("pred_label").eq(lit(1))),
            "fn" => col("ground_truth")
                .eq(lit(1))
                .and(col("pred_label").eq(lit(0))),
            "tn" => col("ground_truth")
                .eq(lit(0))
                .and(col("pred_label").eq(lit(0))),
            _ => return Err(PolarsError::ComputeError("Invalid category".into())),
        };

        let filtered_df = lazy_df
            .filter(filter_condition)
            .select([
                col("ground_truth").cast(DataType::Int32),
                col("pred_label").cast(DataType::Int32),
                col("text"),
            ])
            .collect()?;

        let mut samples = Vec::new();
        let mut row_buffer = polars::frame::row::Row::default(); // Use Row instead of Vec

        for i in 0..filtered_df.height() {
            filtered_df.get_row_amortized(i, &mut row_buffer)?;

            // Debugging: Print the row contents
            println!("Row {}: {:?}", i, row_buffer);

            if row_buffer.0.len() >= 3 {
                let ground_truth = match &row_buffer.0[0] {
                    AnyValue::Int64(val) => *val,
                    _ => {
                        eprintln!("Unexpected type for ground_truth in row {}", i);
                        0 // Default value or handle error
                    }
                };

                let pred_label = match &row_buffer.0[1] {
                    AnyValue::Int64(val) => *val,
                    _ => {
                        eprintln!("Unexpected type for pred_label in row {}", i);
                        0 // Default value or handle error
                    }
                };

                let details = match &row_buffer.0[2] {
                    AnyValue::String(val) => val.to_string(),
                    _ => {
                        eprintln!("Unexpected type for details in row {}", i);
                        String::new() // Default value or handle error
                    }
                };

                samples.push(Sample {
                    id: i,
                    ground_truth,
                    pred_label,
                    details,
                });
            } else {
                eprintln!("Row {} does not have enough elements", i);
            }
        }

        Ok(samples)
    }

    fn get_selected_sample(&self) -> Option<Sample> {
        self.sample_list_state.selected().and_then(|index| {
            self.get_filtered_samples("tp")
                .unwrap_or_else(|_| Vec::new())
                .get(index)
                .cloned()
        })
    }
}

impl SelectedTab {
    fn previous(self) -> Self {
        let current_index: usize = self as usize;
        let previous_index = current_index.saturating_sub(1);
        Self::from_repr(previous_index).unwrap_or(self)
    }

    fn next(self) -> Self {
        let current_index = self as usize;
        let next_index = current_index.saturating_add(1);
        Self::from_repr(next_index).unwrap_or(self)
    }
}
