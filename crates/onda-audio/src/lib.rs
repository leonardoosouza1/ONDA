//! Audio decode/encode + resampling. All I/O funnels through here.

pub mod decode;
pub mod encode;
pub mod resample;

pub use decode::decode_file;
pub use encode::write_wav;
pub use resample::resample_to;
