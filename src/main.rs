use clap::{Arg, Command};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    },
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
use std::fs;
use std::io::stdout;
use std::process;
use strum::{EnumIter, FromRepr, IntoEnumIterator};
fn main() -> Result<(), Box<dyn Error>> {
    // Define the CLI using clap
    let matches = Command::new("MetricWhiz")
        .version("0.1")
        .author("Manpreet Singh")
        .about("Processes a prediction file")
        .arg(
            Arg::new("pred_file_path")
                .help("The path to the prediction csv file. The file should have pred_score, ground_truth and text columns.")
                .required(true)
                .index(1),
        )
        .get_matches();

    // Get the prediction file path from the arguments
    let pred_file_path = matches
        .get_one::<String>("pred_file_path")
        .expect("Prediction file path is required");

    // Check if the file exists
    if !fs::metadata(pred_file_path).is_ok() {
        eprintln!("Error: File '{}' does not exist.", pred_file_path);
        process::exit(1);
    }
    terminal::enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut evaluator = BinaryClsEvaluator::new(&pred_file_path)?;
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
    text: String, // Add any other fields you need
}

#[derive(Clone, Copy, PartialEq, Eq, EnumIter, Debug)]
enum SampleCategory {
    TruePositive,
    TrueNegative,
    FalsePositive,
    FalseNegative,
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
    category_list_state: ListState,
    sample_list_state: ListState,
    selected_category: SampleCategory,
}

impl App {
    fn new(evaluator: BinaryClsEvaluator) -> Self {
        App {
            state: AppState::Running,
            selected_tab: SelectedTab::ReportViewer,
            evaluator,
            sample_list_state: ListState::default(),
            selected_category: SampleCategory::FalsePositive,
            category_list_state: ListState::default(),
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
            .constraints(
                [
                    Constraint::Length(3), // For tabs
                    Constraint::Min(0),    // For main content
                    Constraint::Length(3), // For tooltip
                ]
                .as_ref(),
            )
            .split(size);

        let titles = SelectedTab::iter().map(|tab| {
            Line::from(Span::styled(
                format!("{:?}", tab),
                Style::default()
                    .fg(Color::Cyan)
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

        // Tooltip at the bottom
        let tooltip_text = Text::from(Line::from(Span::styled(
            "Press 'q' to quit, 'n' for next tab, 'p' for previous tab, ←/→ to adjust threshold or select sample category, ↑/↓ to navigate samples.",
            Style::default().fg(Color::Cyan),
        )));
        let tooltip_paragraph = Paragraph::new(tooltip_text)
            .block(Block::default().borders(Borders::ALL).title("Tooltip"))
            .wrap(ratatui::widgets::Wrap { trim: true });
        f.render_widget(tooltip_paragraph, chunks[2]);
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
            .gauge_style(Style::default().fg(Color::Cyan))
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
            .constraints([
                Constraint::Percentage(10), // For category selection
                Constraint::Percentage(40), // For sample list
                Constraint::Percentage(50), // For sample details
            ])
            .split(area);

        // Render the category selection dropdown
        let categories = SampleCategory::iter()
            .map(|category| ListItem::new(format!("{:?}", category)))
            .collect::<Vec<_>>();

        let category_list = List::new(categories)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Select Category"),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">>");

        f.render_stateful_widget(category_list, chunks[0], &mut self.category_list_state);

        // Get filtered samples based on the selected category
        let category_str = match self.selected_category {
            SampleCategory::TruePositive => "tp",
            SampleCategory::TrueNegative => "tn",
            SampleCategory::FalsePositive => "fp",
            SampleCategory::FalseNegative => "fn",
        };

        let samples = self
            .get_filtered_samples(category_str)
            .unwrap_or_else(|_| Vec::new());

        // Render the list of samples
        let sample_list: Vec<_> = samples
            .iter()
            .map(|sample| ListItem::new(format!("{}: {}", sample.id, sample.text)))
            .collect();

        let sample_list_widget = List::new(sample_list)
            .block(Block::default().borders(Borders::ALL).title("Samples"))
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">>");

        f.render_stateful_widget(sample_list_widget, chunks[1], &mut self.sample_list_state);

        // Render the details of the selected sample in the right pane
        if let Some(selected_sample) = self.get_selected_sample(category_str) {
            let sample_details = format!(
                "Ground Truth: {}, Pred Label: {} \n\n{}",
                selected_sample.ground_truth, selected_sample.pred_label, selected_sample.text
            );
            let sample_paragraph = Paragraph::new(sample_details)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Sample Details"),
                )
                .wrap(ratatui::widgets::Wrap { trim: true });
            f.render_widget(sample_paragraph, chunks[2]);
        }
    }

    fn handle_events(&mut self) -> std::io::Result<()> {
        if let Event::Key(KeyEvent {
            code,
            kind: KeyEventKind::Press,
            ..
        }) = event::read()?
        {
            match code {
                KeyCode::Char('q') | KeyCode::Esc => self.quit(),
                KeyCode::Char('p') => self.previous_tab(),
                KeyCode::Char('n') => self.next_tab(),
                KeyCode::Left | KeyCode::Right => {
                    if self.selected_tab == SelectedTab::ReportViewer {
                        // Handle threshold adjustments in the ReportViewer tab
                        self.handle_threshold_events(code);
                    } else if self.selected_tab == SelectedTab::SampleViewer {
                        if code == KeyCode::Left {
                            self.previous_category();
                        } else {
                            self.next_category();
                        }
                    }
                }
                KeyCode::Up | KeyCode::Down => {
                    if self.selected_tab == SelectedTab::SampleViewer {
                        if code == KeyCode::Up {
                            self.previous_sample();
                        } else {
                            self.next_sample();
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn previous_category(&mut self) {
        let i = match self.category_list_state.selected() {
            Some(i) => {
                if i == 0 {
                    SampleCategory::iter().count() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.category_list_state.select(Some(i));
        self.selected_category = SampleCategory::iter().nth(i).unwrap();
    }

    fn next_category(&mut self) {
        let i = match self.category_list_state.selected() {
            Some(i) => {
                if i >= SampleCategory::iter().count() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.category_list_state.select(Some(i));
        self.selected_category = SampleCategory::iter().nth(i).unwrap();
    }

    fn previous_sample(&mut self) {
        let i = match self.sample_list_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.sample_list_state.selected().unwrap_or(0)
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.sample_list_state.select(Some(i));
    }

    fn next_sample(&mut self) {
        let i = match self.sample_list_state.selected() {
            Some(i) => {
                if i >= self
                    .get_filtered_samples("tp")
                    .unwrap_or_else(|_| Vec::new())
                    .len()
                    - 1
                {
                    i
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.sample_list_state.select(Some(i));
    }

    fn get_selected_sample(&self, category: &str) -> Option<Sample> {
        self.sample_list_state.selected().and_then(|index| {
            self.get_filtered_samples(category)
                .unwrap_or_else(|_| Vec::new())
                .get(index)
                .cloned()
        })
    }

    fn get_filtered_samples(&self, category: &str) -> Result<Vec<Sample>, PolarsError> {
        let lazy_df = self.evaluator.pred_df.clone().lazy();

        // Debugging: Print the DataFrame before filtering
        // let df = self.evaluator.pred_df.clone();
        // println!("DataFrame before filtering:\n{:?}", df);

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

        // Debugging: Print the filtered DataFrame
        // println!("Filtered DataFrame:\n{:?}", filtered_df);

        let mut samples = Vec::new();
        let mut row_buffer =
            polars::frame::row::Row::new(vec![AnyValue::Null; filtered_df.width()]); // Initialize Row with correct capacity

        for i in 0..filtered_df.height() {
            filtered_df.get_row_amortized(i, &mut row_buffer)?;

            // Debugging: Print the row contents
            // println!("Row {}: {:?}", i, row_buffer);

            if row_buffer.0.len() >= 3 {
                let ground_truth = match &row_buffer.0[0] {
                    AnyValue::Int32(val) => *val as i64,
                    _ => {
                        eprintln!("Unexpected type for ground_truth in row {}", i);
                        0 // Default value or handle error
                    }
                };

                let pred_label = match &row_buffer.0[1] {
                    AnyValue::Int32(val) => *val as i64,
                    _ => {
                        eprintln!("Unexpected type for pred_label in row {}", i);
                        0 // Default value or handle error
                    }
                };

                let text = match &row_buffer.0[2] {
                    AnyValue::String(val) => val.to_string(),
                    _ => {
                        eprintln!("Unexpected type for text in row {}", i);
                        String::new() // Default value or handle error
                    }
                };

                samples.push(Sample {
                    id: i,
                    ground_truth,
                    pred_label,
                    text,
                });
            } else {
                eprintln!("Row {} does not have enough elements", i);
            }
        }

        Ok(samples)
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
