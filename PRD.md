# MAVEN Product Requirements Document

## 1. Product Overview

MAVEN (Motion-Aware Visual Evaluation Network) is a web-based ASD screening prototype that analyzes raw toddler video to produce a binary prediction, supporting visual explanations for screening review.

The product consists of:
- A landing page that introduces the system and provides a demo preview.
- A screening UI that accepts a single video upload.
- A backend inference pipeline that samples frames, runs the video model, and returns ASD/TD probability, confidence, frame importance, and timing data.

This product is intended for research and AI-assisted screening only. It is not a clinical diagnostic tool.

## 2. Problem Statement

Clinicians and researchers need a lightweight way to review toddler motion patterns from raw video without requiring manual pose labeling or specialist preprocessing. The current workflow should make it easy to upload a video, run inference, and inspect model evidence quickly.

## 3. Goals

- Provide a simple browser-based ASD screening flow from uploaded video.
- Deliver a prediction with probability and confidence in one pass.
- Show interpretable evidence using frame attention and per-frame feature summaries.
- Keep the system fast enough for interactive use on a single machine with optional GPU acceleration.
- Preserve a clean modular architecture so the model, preprocessing, and UI can evolve independently.

## 4. Non-Goals

- Clinical diagnosis or treatment recommendations.
- Multi-patient case management or longitudinal record storage.
- Multi-video batch screening in the current UI.
- Manual pose annotation, skeleton editing, or training-time labeling tools in the frontend.
- Regulatory certification, medical device clearance, or production clinical deployment.

## 5. Target Users

- Researcher: Evaluates model behavior, checks attention patterns, and compares experiments.
- Clinician reviewer: Uses the tool as a screening aid and inspects output explanations.
- Developer / ML engineer: Validates the inference path, model contract, and frontend integration.

## 6. Core User Journey

1. User opens the landing page and optionally watches the demo video.
2. User opens the screening page.
3. User uploads a toddler video in MP4, AVI, or MOV format.
4. System previews the file locally in the browser.
5. User starts screening.
6. Backend extracts metadata, samples frames, encodes frames, applies temporal modeling, and returns results.
7. Frontend renders:
   - ASD or TD label
   - ASD and TD probabilities
   - Confidence score
   - Top attention frames
   - Frame thumbnails
   - CNN feature energy chart
   - Attention weight chart
   - Timing breakdown

## 7. Functional Requirements

### 7.1 Landing Page
- Display product branding and purpose.
- Show a demo video if available.
- Provide a clear entry point to the screening page.

### 7.2 Screening Page
- Accept one uploaded video at a time.
- Support drag-and-drop and file picker upload.
- Show a local preview before inference.
- Display basic video metadata such as duration, resolution, frame count, and FPS.
- Provide a visible run-screening action.

### 7.3 Inference API
- Accept a multipart form upload with a `video` field.
- Return an error if the file is missing or cannot be processed.
- Load the model checkpoint once and reuse it across requests.
- Run inference on the uploaded file and return a JSON response.

### 7.4 Model Output
- Return a binary label: ASD or TD.
- Return ASD probability and complementary TD probability.
- Return confidence for the predicted class.
- Return frame weights and top frame indices.
- Return frame thumbnails for the sampled frames.
- Return per-stage timing so the UI can show performance.

### 7.5 Explainability UI
- Highlight the most important frames.
- Render charts for CNN feature energy and attention weights.
- Present a short natural-language explanation using the top frames and attention peak.

## 8. Model and Data Requirements

The deployed screening flow currently expects:
- Input type: raw video file.
- Sampling path: uniformly sampled frames.
- Inference tensor shape: `(1, 16, 3, 96, 96)` in the running app.
- Output type: probability and frame-level attention metadata.

The architecture should remain modular so the frame backbone, temporal encoder, and classification head can be replaced without changing the user flow.

## 9. UX Requirements

- The landing page should feel like a product entry point, not a training console.
- The screening page should communicate progress through staged visualization.
- Result presentation should be readable on desktop and usable on smaller screens.
- Error states must be visible and actionable.
- The interface should clearly state that the tool is a screening aid, not a diagnosis.

## 10. Non-Functional Requirements

- Performance: Show a result in a reasonable interactive time on a local machine, with GPU acceleration when available.
- Reliability: Clean up temporary uploaded files after each request.
- Security: Limit upload size and avoid storing user videos beyond the active request.
- Maintainability: Keep model, preprocessing, and frontend concerns separated.
- Portability: Run locally with the provided Python environment and Flask server.

## 11. Success Metrics

- Successful upload-to-result flow without manual intervention.
- Correct rendering of model metadata, charts, thumbnails, and prediction output.
- Low rate of frontend/backend contract errors.
- Interactive latency that remains practical for demo and review usage.
- Reproducible inference from the same video and checkpoint.

## 12. Current Implementation Snapshot

- Frontend routes:
  - `/` landing page
  - `/model` screening page
- Backend routes:
  - `/predict` video inference
  - `/model_info` checkpoint metadata
- Model family:
  - MobileNetV3-Small frame encoder
  - Factorized temporal transformer
  - Temporal self-attention
  - MLP classification head
- Current checkpoint file:
  - `checkpoints/video_model_best.pth`

## 13. Risks and Constraints

- The repository is positioned for screening research, not clinical use.
- The frontend depends on external browser assets for some visual effects, which may fail in restricted environments.
- The demo video asset may be absent in some setups.
- Model and documentation parameters must stay aligned to avoid confusion between archived and deployed settings.

## 14. Acceptance Criteria

- A user can open the landing page and navigate to the screening page.
- A user can upload a supported video and receive a result.
- The UI shows probabilities, confidence, top frames, and timing breakdown.
- The backend returns structured JSON with the fields expected by the frontend.
- The app runs locally from the Flask entrypoint without additional manual wiring.

## 15. Future Enhancements

- Optional batch screening for multiple videos.
- Stronger explainability views, including frame-by-frame comparison and threshold tuning.
- Better offline asset handling for the landing page background and demo video.
- Alignment of documented input size and frame count with the deployed checkpoint and preprocessing.
- Potential multi-stream extensions such as skeleton or optical flow fusion.

## 16. Open Questions

- Should the deployed inference settings be standardized at 16 frames and 96x96, or should the training and documentation be updated to a 30-frame 112x112 configuration?
- Should the landing page background assets be bundled locally to remove the external dependency?
- Should the product expose threshold tuning or calibration settings in the UI for research review?
