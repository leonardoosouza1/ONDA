//! Audio downloader — Rust wrappers around Python helpers (uv + yt-dlp).

pub mod archive;
pub mod runner;
pub mod youtube;

pub use archive::{search_cc_audio, ArchiveHit};
pub use runner::{download, DownloadError, DownloadInfo, DownloadOptions, Format};
pub use youtube::{yt_search, YtHit};
