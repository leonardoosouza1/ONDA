//! archive.org catalog search wrapper (calls `tools/search_archive.py`).

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::runner::DownloadError;

#[derive(Debug, Clone, Deserialize)]
pub struct ArchiveHit {
    pub url: String,
    pub identifier: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub creator: Option<serde_json::Value>,
    #[serde(default)]
    pub year: Option<serde_json::Value>,
}

fn locate_search_script() -> Result<PathBuf, DownloadError> {
    if let Ok(p) = std::env::var("ONDA_SEARCH_SCRIPT") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let cwd = PathBuf::from("tools/search_archive.py");
    if cwd.exists() {
        return Ok(cwd.canonicalize()?);
    }
    let start = std::env::var("CARGO_MANIFEST_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut cur: Option<&Path> = Some(&start);
    while let Some(dir) = cur {
        let cand = dir.join("tools/search_archive.py");
        if cand.exists() {
            return Ok(cand);
        }
        cur = dir.parent();
    }
    Err(DownloadError::ScriptMissing)
}

/// Query archive.org for CC audio items in the netlabels collection.
pub async fn search_cc_audio(
    subject: &str,
    limit: u32,
) -> Result<Vec<ArchiveHit>, DownloadError> {
    let script = locate_search_script()?;
    info!(subject, limit, script=?script, "archive.org search");

    let mut child = Command::new("uv")
        .arg("run")
        .arg("--script")
        .arg(&script)
        .arg("--subject")
        .arg(subject)
        .arg("--limit")
        .arg(limit.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let stderr_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            warn!(target: "onda_download::archive", "{line}");
        }
    });

    let mut hits = Vec::new();
    let mut lines = BufReader::new(stdout).lines();
    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        debug!(target: "onda_download::archive", "hit: {trimmed}");
        match serde_json::from_str::<ArchiveHit>(trimmed) {
            Ok(h) => hits.push(h),
            Err(e) => warn!("unparseable hit: {trimmed} ({e})"),
        }
    }

    let status = child.wait().await?;
    let _ = stderr_task.await;

    if !status.success() {
        return Err(DownloadError::Script {
            code: status.code().unwrap_or(-1),
            stderr: "archive.org search failed".into(),
        });
    }

    Ok(hits)
}
