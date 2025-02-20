mod cli;
mod errors;

use std::env;

use tracing_appender::rolling;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

fn main() {
    // Setup tracing
    if let Ok(log_dir) = env::var("ACVM_LOG_DIR") {
        let debug_file = rolling::daily(log_dir, "acvm-log");
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::ACTIVE)
            .with_writer(debug_file)
            .with_ansi(false)
            .with_env_filter(EnvFilter::from_default_env())
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::ACTIVE)
            .with_ansi(true)
            .with_env_filter(EnvFilter::from_env("NOIR_LOG"))
            .init();
    }

    if let Err(report) = cli::start_cli() {
        eprintln!("{report}");
        std::process::exit(1);
    }
}
