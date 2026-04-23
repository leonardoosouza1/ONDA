//! YouTube search / channel enumeration via `tools/yt_search.py`.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::runner::DownloadError;

#[derive(Debug, Clone, Deserialize)]
pub struct YtHit {
    pub url: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
}

fn locate_script() -> Result<PathBuf, DownloadError> {
    if let Ok(p) = std::env::var("ONDA_YT_SEARCH_SCRIPT") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let cwd = PathBuf::from("tools/yt_search.py");
    if cwd.exists() {
        return Ok(cwd.canonicalize()?);
    }
    let start = std::env::var("CARGO_MANIFEST_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut cur: Option<&Path> = Some(&start);
    while let Some(dir) = cur {
        let cand = dir.join("tools/yt_search.py");
        if cand.exists() {
            return Ok(cand);
        }
        cur = dir.parent();
    }
    Err(DownloadError::ScriptMissing)
}

/// Search YouTube — `query` can be free text (ytsearchN:...) or a channel/
/// playlist URL (enumerated via yt-dlp).
pub async fn yt_search(
    query: &str,
    limit: u32,
    min_sec: u32,
    max_sec: u32,
) -> Result<Vec<YtHit>, DownloadError> {
    let script = locate_script()?;
    info!(query, limit, min_sec, max_sec, script=?script, "yt-dlp search");

    let mut child = Command::new("uv")
        .arg("run")
        .arg("--script")
        .arg(&script)
        .arg("--query")
        .arg(query)
        .arg("--limit")
        .arg(limit.to_string())
        .arg("--min-sec")
        .arg(min_sec.to_string())
        .arg("--max-sec")
        .arg(max_sec.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let stderr_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            warn!(target: "onda_download::youtube", "{line}");
        }
    });

    let mut hits = Vec::new();
    let mut lines = BufReader::new(stdout).lines();
    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        debug!(target: "onda_download::youtube", "hit: {trimmed}");
        match serde_json::from_str::<YtHit>(trimmed) {
            Ok(h) => hits.push(h),
            Err(e) => warn!("unparseable: {trimmed} ({e})"),
        }
    }

    let status = child.wait().await?;
    let _ = stderr_task.await;

    if !status.success() {
        return Err(DownloadError::Script {
            code: status.code().unwrap_or(-1),
            stderr: "yt_search.py failed".into(),
        });
    }
    Ok(hits)
}
