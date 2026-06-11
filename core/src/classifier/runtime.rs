//! ONNX Runtime wrapper and shared utilities.

use ndarray::{Array, ArrayD};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

/// Paths to the AI classification models.
#[derive(Clone, Debug)]
pub struct ModelPaths {
    pub nsfw_model: PathBuf,
    pub tagging_model: PathBuf,
}

impl Default for ModelPaths {
    fn default() -> Self {
        // Look for models in .vendor/models/ relative to workspace root
        // This assumes the binary is run from the workspace root
        let vendor_models = PathBuf::from(".vendor/models");

        Self {
            nsfw_model: vendor_models.join("nsfw-inception-v3.onnx"),
            tagging_model: vendor_models.join("mobilenetv2-12.onnx"),
        }
    }
}

impl ModelPaths {
    /// Create model paths with a custom base directory.
    pub fn with_base_dir(base: impl AsRef<Path>) -> Self {
        let base = base.as_ref();
        Self {
            nsfw_model: base.join("nsfw-inception-v3.onnx"),
            tagging_model: base.join("mobilenetv2-12.onnx"),
        }
    }

    /// Check if all required models exist.
    pub fn validate(&self) -> Result<(), ClassifierError> {
        if !self.nsfw_model.exists() {
            return Err(ClassifierError::ModelNotFound(self.nsfw_model.clone()));
        }
        if !self.tagging_model.exists() {
            return Err(ClassifierError::ModelNotFound(self.tagging_model.clone()));
        }
        Ok(())
    }
}

/// Default ImageNet normalization mean (RGB order).
pub const IMAGE_NET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Default ImageNet normalization standard deviation (RGB order).
pub const IMAGE_NET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Load an ONNX session from a model file.
pub fn load_session(model_path: &Path) -> Result<Session, ClassifierError> {
    if !model_path.exists() {
        return Err(ClassifierError::ModelNotFound(model_path.to_path_buf()));
    }

    // Read model file into memory
    let model_bytes = std::fs::read(model_path).map_err(|e| {
        ClassifierError::Processing(format!("failed to read model file: {}", e))
    })?;

    Session::builder()
        .map_err(ClassifierError::Ort)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ClassifierError::Ort)?
        .with_intra_threads(4)
        .map_err(ClassifierError::Ort)?
        .commit_from_memory(&model_bytes)
        .map_err(ClassifierError::Ort)
}

/// Preprocess an image for model inference.
/// Resizes and normalizes to the expected input format.
#[allow(dead_code)]
pub fn preprocess_image(
    image_path: &Path,
    target_size: (i32, i32),
    normalize: bool,
) -> Result<ArrayD<f32>, ClassifierError> {
    preprocess_image_with_layout(
        image_path,
        target_size,
        normalize,
        "NCHW",
        IMAGE_NET_MEAN,
        IMAGE_NET_STD,
        true,
    )
}

/// Preprocess an image with configurable layout.
pub fn preprocess_image_with_layout(
    image_path: &Path,
    target_size: (i32, i32),
    normalize: bool,
    layout: &str,
    mean: [f32; 3],
    std: [f32; 3],
    batch_dim: bool,
) -> Result<ArrayD<f32>, ClassifierError> {
    use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, Size, CV_32FC3};
    use opencv::imgcodecs;
    use opencv::imgproc;

    // Load image
    let image = imgcodecs::imread(
        image_path
            .to_str()
            .ok_or_else(|| ClassifierError::InvalidPath(image_path.to_path_buf()))?,
        imgcodecs::IMREAD_COLOR,
    )
    .map_err(ClassifierError::OpenCv)?;

    if image.empty() {
        return Err(ClassifierError::InvalidPath(image_path.to_path_buf()));
    }

    // Resize to target size
    let mut resized = Mat::default();
    imgproc::resize(
        &image,
        &mut resized,
        Size::new(target_size.0, target_size.1),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .map_err(ClassifierError::OpenCv)?;

    // Convert to float32
    let mut float_image = Mat::default();
    resized
        .convert_to(&mut float_image, CV_32FC3, 1.0 / 255.0, 0.0)
        .map_err(ClassifierError::OpenCv)?;

    // Extract data and convert to ndarray
    // OpenCV is BGR, HWC format -> we need RGB, NCHW for ONNX
    let rows = float_image.rows() as usize;
    let cols = float_image.cols() as usize;
    let channels = 3usize;

    let data: Vec<f32> = float_image
        .data_typed::<opencv::core::Vec3f>()
        .map_err(ClassifierError::OpenCv)?
        .iter()
        .flat_map(|pixel| {
            // BGR -> RGB
            [pixel[2], pixel[1], pixel[0]]
        })
        .collect();

    // Create HWC array
    let hwc = Array::from_shape_vec((rows, cols, channels), data)
        .map_err(|e| ClassifierError::Processing(e.to_string()))?;

    // Handle different layouts
    if layout == "NHWC" {
        if batch_dim {
            // Add batch dimension -> NHWC [1, H, W, C]
            let nhwc = hwc.insert_axis(ndarray::Axis(0));

            if normalize {
                let mut normalized = nhwc.into_owned();
                for c in 0..3 {
                    normalized
                        .slice_mut(ndarray::s![0, .., .., c])
                        .mapv_inplace(|v| (v - mean[c]) / std[c]);
                }
                Ok(normalized.into_dyn())
            } else {
                Ok(nhwc.into_owned().into_dyn())
            }
        } else {
            // No batch dimension -> HWC [H, W, C]
            if normalize {
                let mut normalized = hwc.into_owned();
                for c in 0..3 {
                    normalized
                        .slice_mut(ndarray::s![.., .., c])
                        .mapv_inplace(|v| (v - mean[c]) / std[c]);
                }
                Ok(normalized.into_dyn())
            } else {
                Ok(hwc.into_owned().into_dyn())
            }
        }
    } else {
        // NCHW format (default, PyTorch-style)
        // Transpose HWC -> CHW
        let chw = hwc.permuted_axes([2, 0, 1]);

        if batch_dim {
            // Add batch dimension -> NCHW [1, C, H, W]
            let nchw = chw.insert_axis(ndarray::Axis(0));

            if normalize {
                let mut normalized = nchw.into_owned();
                for c in 0..3 {
                    normalized
                        .slice_mut(ndarray::s![0, c, .., ..])
                        .mapv_inplace(|v| (v - mean[c]) / std[c]);
                }
                Ok(normalized.into_dyn())
            } else {
                Ok(nchw.into_owned().into_dyn())
            }
        } else {
            // No batch dimension -> CHW [C, H, W]
            if normalize {
                let mut normalized = chw.into_owned();
                for c in 0..3 {
                    normalized
                        .slice_mut(ndarray::s![c, .., ..])
                        .mapv_inplace(|v| (v - mean[c]) / std[c]);
                }
                Ok(normalized.into_dyn())
            } else {
                Ok(chw.into_owned().into_dyn())
            }
        }
    }
}

/// Softmax function for converting logits to probabilities.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|x| x / sum).collect()
}

/// Sigmoid function for converting a single logit to probability.
/// Used for multi-label classification where each output is independent.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Errors that can occur during classification.
#[derive(Debug)]
pub enum ClassifierError {
    ModelNotFound(PathBuf),
    InvalidPath(PathBuf),
    Ort(ort::Error),
    OpenCv(opencv::Error),
    Processing(String),
}

impl Display for ClassifierError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound(path) => {
                write!(f, "model not found: {}", path.display())
            }
            Self::InvalidPath(path) => {
                write!(f, "invalid image path: {}", path.display())
            }
            Self::Ort(e) => write!(f, "ONNX runtime error: {}", e),
            Self::OpenCv(e) => write!(f, "OpenCV error: {}", e),
            Self::Processing(msg) => write!(f, "processing error: {}", msg),
        }
    }
}

impl std::error::Error for ClassifierError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenCv(e) => Some(e),
            _ => None,
        }
    }
}
