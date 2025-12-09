# Classification System Analysis

This document captures findings from comparative testing of single-model vs ensemble classification modes, along with recommendations for improvements.

## Test Setup

- **Dataset:** 49 images from `D:\Wallpapers\Triage\Landscape Classification Test`
- **Single Mode:** `taufiqdp-mobilenetv4-nsfw` (moderation) + `smilingwolf-wd-v1-4-convnextv2-tagger-v2` (tagging)
- **Ensemble Mode:** 3 moderation models + 3 tagging models (ConvNeXtV2, JoyTag, CL Tagger)
- **Date:** 2025-12-09

## Results Summary

| Metric | Ensemble Mode | Single Mode |
|--------|--------------|-------------|
| Files Processed | 49 | 49 |
| Safe | 37 | 37 |
| Sensitive | 10 | 12 |
| Restricted | 2 | 0 |
| Avg Tags/File | ~5 | ~2 |
| Tag Quality | Semantic (1girl, nipples, etc.) | Generic (general, monochrome) |

### Key Observations

1. **Ensemble produces meaningful semantic tags** like `1girl`, `solo`, `nipples`, `breasts`
2. **Single mode produces only rating/generic tags** like `general`, `monochrome`, `greyscale`
3. **Ensemble shows confidence values >100%** (e.g., `1girl (303%)`)
4. **Moderation tiers inconsistent with tag content** (images with `nipples` tag marked as "Safe")

---

## Issues Identified

### Issue 1: Missing Sigmoid Application (CRITICAL)

**Location:** `core/src/classifier/tagging.rs:235-237`

**Code:**
```rust
let probabilities = if self.config.multi_label {
    // Multi-label: use raw sigmoid outputs directly
    logits.clone()  // ASSUMES sigmoid is already applied
} else {
    // Single-label: apply softmax if needed
    ...
}
```

**Problem:** The code assumes multi-label models (WD taggers, JoyTag, CL Tagger) output sigmoid-activated probabilities in [0, 1]. However, if an ONNX model outputs raw logits (pre-sigmoid), values can exceed 1.0.

**Evidence:** Ensemble results show `1girl (303%)` which means confidence = 3.03, indicating raw logits are being used as probabilities.

**Impact:**
- Confidence values are meaningless (>100%)
- Threshold filtering (`min_confidence: 0.6`) doesn't work correctly
- Tag ranking may be affected

**Recommendation:** Add explicit sigmoid activation for multi-label models:

```rust
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

let probabilities = if self.config.multi_label {
    // Apply sigmoid to convert logits to probabilities
    logits.iter().map(|&x| sigmoid(x)).collect()
} else {
    // Single-label: apply softmax if needed
    ...
}
```

**Alternative:** Add a `needs_sigmoid` flag to model configuration for models that don't include sigmoid in their ONNX graph.

---

### Issue 2: CL Tagger JSON Format Not Supported

**Location:** `core/src/classifier/labels.rs:207-239`

**Problem:** The CL Tagger uses an index-mapped JSON format:

```json
{
  "0": {"tag": "general", "category": "Rating"},
  "1": {"tag": "sensitive", "category": "Rating"},
  "4": {"tag": "1girl", "category": "General"},
  ...
}
```

But `load_labels_from_json()` only supports:
- Array format: `["label1", "label2", ...]`
- Object with labels field: `{"labels": ["label1", ...]}`

**Impact:** CL Tagger's 42,163 labels may not be loading correctly, causing index mismatches between model outputs and label names.

**Recommendation:** Add support for index-mapped format:

```rust
// Try parsing as index-mapped object
#[derive(serde::Deserialize)]
struct TagEntry {
    tag: String,
    category: Option<String>,
}

if let Ok(map) = serde_json::from_str::<HashMap<String, TagEntry>>(&content) {
    // Sort by index and extract tags
    let mut entries: Vec<(usize, String)> = map
        .into_iter()
        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v.tag)))
        .collect();
    entries.sort_by_key(|(i, _)| *i);

    let labels: Vec<String> = entries.into_iter().map(|(_, tag)| tag).collect();
    if !labels.is_empty() {
        return Ok(labels);
    }
}
```

**Bonus:** Extract category information for tag categorization (Rating, General, Character, etc.).

---

### Issue 3: Tag Confidence Threshold Too Aggressive

**Location:** `core/src/classifier/tagging.rs:19`

**Code:**
```rust
pub const DEFAULT_MIN_TAG_CONFIDENCE: f32 = 0.6;  // 60%
```

**Problem:** A 60% threshold is too high for multi-label models with thousands of tags:
- WD taggers have 9,999+ tags where probability mass is spread thin
- Rating tags (general, sensitive) often dominate with high confidence
- Useful content tags (1girl, solo, blonde_hair) may score 40-59% and get filtered

**Evidence:** Single mode results show only 2-3 tags per image, mostly ratings.

**Recommendation:**

1. **Lower default threshold to 0.35** for better tag coverage
2. **Make threshold configurable per-model** in TOML:
   ```toml
   [[tagging_models]]
   name = "smilingwolf-wd-v1-4-convnextv2-tagger-v2"
   min_confidence = 0.35
   ```
3. **Consider separate thresholds** for rating vs content tags
4. **For ensemble mode**, use lower threshold (0.30) since averaging naturally filters noise

---

### Issue 4: Ensemble Tag Merging Strategy

**Location:** `core/src/classifier/ensemble.rs:128-168`

**Current Logic:**
```rust
let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
```

**Behavior:** Averages confidence only among models that detected the tag. If 1 of 3 models detects `rare_tag` at 0.9, average is 0.9 (not 0.3).

**Problem:** No penalty for low model agreement. A tag detected by 1 model with high confidence ranks equally with a tag detected by all 3 models.

**Recommendation:** Add agreement weighting:

```rust
pub fn merge_tags(
    all_tags: Vec<Vec<ImageTag>>,
    min_confidence: f32,
    total_models: usize,  // New parameter
) -> Vec<ImageTag> {
    // ... existing grouping logic ...

    let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;

    // Weight by model agreement (0.5 base + 0.5 * agreement ratio)
    let agreement_ratio = confidences.len() as f32 / total_models as f32;
    let weighted_confidence = avg_confidence * (0.5 + 0.5 * agreement_ratio);

    if weighted_confidence >= min_confidence {
        // ... create tag with weighted_confidence ...
    }
}
```

**Alternative strategies:**
- **Require minimum agreement:** Only include tags detected by ≥2 models
- **Confidence boosting:** Multiply confidence by agreement ratio
- **Intersection mode:** Only include tags detected by ALL models (strictest)

---

### Issue 5: Moderation Pipeline Ignores Tag Evidence

**Location:** `core/src/classifier/mod.rs:328-337`

**Current Logic:**
```rust
pub fn classify(&mut self, image_path: &Path) -> Result<ClassificationResult, ClassifierError> {
    let moderation = self.moderate(image_path)?;  // Independent
    let tags = self.tag(image_path, ...)?;        // Independent
    Ok(ClassificationResult { moderation, tags }) // No cross-reference
}
```

**Problem:** Moderation tier is determined solely by moderation models (GantMan categories). Tags containing explicit content indicators (`nipples`, `nude`, `pussy`, `sex`) don't influence the tier.

**Evidence:** Images with `nipples (175%)` and `nude (103%)` tags are classified as "Safe" by moderation.

**Recommendation:** Add tag-based tier adjustment:

```rust
/// NSFW-indicative tags and their severity weights
const NSFW_TAG_WEIGHTS: &[(&str, f32)] = &[
    ("explicit", 1.0),
    ("nude", 0.9),
    ("pussy", 0.95),
    ("penis", 0.95),
    ("sex", 1.0),
    ("nipples", 0.6),
    ("breasts", 0.3),  // Lower weight - context dependent
    ("topless", 0.7),
    ("bottomless", 0.8),
];

fn adjust_tier_from_tags(tier: ModerationTier, tags: &[ImageTag]) -> ModerationTier {
    let nsfw_evidence: f32 = tags.iter()
        .filter_map(|tag| {
            NSFW_TAG_WEIGHTS.iter()
                .find(|(name, _)| tag.name.contains(name))
                .map(|(_, weight)| tag.confidence * weight)
        })
        .sum();

    // Escalate tier based on cumulative evidence
    match (tier, nsfw_evidence) {
        (ModerationTier::Safe, e) if e > 0.5 => ModerationTier::Sensitive,
        (ModerationTier::Safe, e) if e > 1.0 => ModerationTier::Mature,
        (ModerationTier::Sensitive, e) if e > 1.0 => ModerationTier::Mature,
        (t, e) if e > 2.0 => ModerationTier::Restricted,
        (t, _) => t,
    }
}
```

**Integration point:** Call after classification in `classify()` method.

---

### Issue 6: Max Tags Limit Too Restrictive for Ensemble

**Location:** `core/src/scanner.rs:351`

**Code:**
```rust
match classifier.tag(path, MAX_TAGS_PER_IMAGE) {  // MAX_TAGS_PER_IMAGE = 5
```

**Problem:** Ensemble mode naturally produces more diverse tags from multiple models, but the 5-tag limit truncates valuable information.

**Recommendation:**

1. **Increase limit for ensemble mode:**
   ```rust
   let max_tags = if classifier.is_ensemble_mode() { 15 } else { 5 };
   ```

2. **Make configurable in TOML:**
   ```toml
   [classification]
   max_tags_single = 5
   max_tags_ensemble = 15
   ```

3. **Dynamic based on model count:**
   ```rust
   let max_tags = 5 * classifier.tagging_model_count();
   ```

---

## Priority Matrix

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| **P0** | Missing sigmoid | Low | Critical - fixes >100% values |
| **P0** | CL Tagger JSON | Medium | Critical - enables 42k tags |
| **P1** | Tag threshold | Low | High - better tag coverage |
| **P1** | Tag→Moderation | Medium | High - accurate content rating |
| **P2** | Agreement weighting | Low | Medium - better ensemble quality |
| **P2** | Max tags config | Low | Medium - richer output |

---

## Verification Steps

### Confirm Sigmoid Issue

Add debug logging to `tagging.rs`:

```rust
// After getting logits, before processing
let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
eprintln!("[DEBUG] Logit range: [{:.3}, {:.3}]", min_logit, max_logit);

if max_logit > 1.0 || min_logit < 0.0 {
    eprintln!("[WARN] Model outputs appear to be raw logits, not probabilities!");
}
```

### Confirm CL Tagger Label Loading

```rust
// In load_labels_from_json, add fallback logging
eprintln!("[DEBUG] JSON structure: first 100 chars = {}", &content[..100.min(content.len())]);
```

---

## Configuration Recommendations

### Optimized Single Model Config

```toml
models_dir = ".vendor/models"
ort_library = ".vendor/onnxruntime/lib/onnxruntime.dll"

moderation_model = "taufiqdp-mobilenetv4-nsfw"
tagging_model = "smilingwolf-wd-v1-4-convnextv2-tagger-v2"

[tagging]
min_confidence = 0.35  # Lower threshold for single model
max_tags = 10
```

### Optimized Ensemble Config

```toml
models_dir = ".vendor/models"
ort_library = ".vendor/onnxruntime/lib/onnxruntime.dll"

moderation_models = [
    "spiele-nsfw-image-detector",
    "taufiqdp-mobilenetv4-nsfw",
    "vladmandic-nudenet",
]

tagging_models = [
    "smilingwolf-wd-v1-4-convnextv2-tagger-v2",
    "fancyfeast-joytag",
    "cella110n-cl-tagger",
]

[tagging]
min_confidence = 0.30  # Lower for ensemble averaging
max_tags = 15
apply_sigmoid = true   # Force sigmoid application

[ensemble]
require_agreement = 2  # Minimum models that must detect a tag
weight_by_agreement = true

[moderation]
use_tag_evidence = true  # Cross-reference tags for tier adjustment
```

---

## Next Steps

1. **Implement P0 fixes** (sigmoid, JSON format)
2. **Run comparison tests** with fixed code
3. **Tune thresholds** based on results
4. **Implement tag→moderation integration**
5. **Add configuration options** to TOML schema

---

## References

- `core/src/classifier/tagging.rs` - Tagging classifier implementation
- `core/src/classifier/ensemble.rs` - Ensemble merging logic
- `core/src/classifier/labels.rs` - Label loading strategies
- `core/src/classifier/moderation/mod.rs` - Moderation tier determination
- `core/src/scanner.rs` - Scanner integration
