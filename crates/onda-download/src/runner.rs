//! Subprocess runner for `tools/download.py` (uv + yt-dlp).

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, info, warn};

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("script failed (exit {code}): {stderr}")]
    Script { code: i32, stderr: String },
    #[error("yt-dlp error: {0}")]
    YtDlp(String),
    #[error("could not locate tools/download.py — set ONDA_DOWNLOAD_SCRIPT or run from repo root")]
    ScriptMissing,
    #[error("no JSON result line from download.py")]
    NoResult,
    #[error("parse: {0}")]
    Parse(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy)]
pub enum Format {
    Mp3,
    M4a,
    BestAudio,
}

impl Format {
    fn as_arg(&self) -> &'static str {
        match self {
            Format::Mp3 => "mp3",
            Format::M4a => "m4a",
            Format::BestAudio => "bestaudio",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub format: Format,
    pub max_duration_secs: u32,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            format: Format::Mp3,
            max_duration_secs: 900,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ScriptOk {
    #[allow(dead_code)]
    ok: bool,
    path: String,
    title: Option<String>,
    duration: Option<f64>,
    uploader: Option<String>,
    extractor: Option<String>,
    webpage_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ScriptErr {
    #[allow(dead_code)]
    ok: bool,
    error: String,
}

#[derive(Debug, Clone)]
pub struct DownloadInfo {
    pub path: PathBuf,
    pub title: Option<String>,
    pub duration: Option<f64>,
    pub uploader: Option<String>,
    pub extractor: Option<String>,
    pub webpage_url: Option<String>,
}

/// Locate `tools/download.py`. Checks, in order:
/// 1. `ONDA_DOWNLOAD_SCRIPT` env var
/// 2. `./tools/download.py` (relative to CWD)
/// 3. Walking up from `CARGO_MANIFEST_DIR` looking for `tools/download.py`
fn locate_script() -> Result<PathBuf, DownloadError> {
    if let Ok(p) = std::env::var("ONDA_DOWNLOAD_SCRIPT") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let cwd_rel = PathBuf::from("tools/download.py");
    if cwd_rel.exists() {
        return Ok(cwd_rel.canonicalize()?);
    }
    // Walk up from CARGO_MANIFEST_DIR or current exe dir
    let start = std::env::var("CARGO_MANIFEST_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())))
        .unwrap_or_else(|| PathBuf::from("."));
    let mut cur: Option<&Path> = Some(&start);
    while let Some(dir) = cur {
        let cand = dir.join("tools/download.py");
        if cand.exists() {
            return Ok(cand);
        }
        cur = dir.parent();
    }
    Err(DownloadError::ScriptMissing)
}

/// Download one URL to `out_dir`. Returns metadata on success.
///
/// Runs `uv run tools/download.py <url> <out_dir> --format <fmt>` and parses
/// the final JSON line from stdout. Stderr is forwarded to tracing::warn.
pub async fn download(
    url: &str,
    out_dir: impl AsRef<Path>,
    opts: &DownloadOptions,
) -> Result<DownloadInfo, DownloadError> {
    let script = locate_script()?;
    let out_dir = out_dir.as_ref();
    tokio::fs::create_dir_all(out_dir).await?;

    info!(url, script=?script, out=?out_dir, "spawning uv run download.py");

    let mut child = Command::new("uv")
        .arg("run")
        .arg("--script")
        .arg(&script)
        .arg(url)
        .arg(out_dir)
        .arg("--format")
        .arg(opts.format.as_arg())
        .arg("--max-duration")
        .arg(opts.max_duration_secs.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let stderr_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        let mut buf = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            warn!(target: "onda_download::py", "{line}");
            buf.push_str(&line);
            buf.push('\n');
        }
        buf
    });

    let mut last_line: Option<String> = None;
    let mut lines = BufReader::new(stdout).lines();
    while let Some(line) = lines.next_line().await? {
        debug!(target: "onda_download::py", "stdout: {line}");
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            last_line = Some(trimmed.to_string());
        }
    }

    let status = child.wait().await?;
    let stderr_buf = stderr_task.await.unwrap_or_default();

    let Some(json_line) = last_line else {
        if !status.success() {
            return Err(DownloadError::Script {
                code: status.code().unwrap_or(-1),
                stderr: stderr_buf,
            });
        }
        return Err(DownloadError::NoResult);
    };

    // Try ok variant first, then err variant.
    if let Ok(ok) = serde_json::from_str::<ScriptOk>(&json_line) {
        if ok.ok {
            return Ok(DownloadInfo {
                path: PathBuf::from(ok.path),
                title: ok.title,
                duration: ok.duration,
                uploader: ok.uploader,
                extractor: ok.extractor,
                webpage_url: ok.webpage_url,
            });
        }
    }
    if let Ok(err) = serde_json::from_str::<ScriptErr>(&json_line) {
        return Err(DownloadError::YtDlp(err.error));
    }

    Err(DownloadError::Script {
        code: status.code().unwrap_or(-1),
        stderr: format!("unparseable output: {json_line}\n---stderr---\n{stderr_buf}"),
    })
}
