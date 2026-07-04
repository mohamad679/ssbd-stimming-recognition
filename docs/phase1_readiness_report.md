# Phase 1 Readiness Report

## Scope and framing

Phase 1 is complete as a tested, privacy-conscious scaffold for proxy motor-behavior recognition. This project is not an autism diagnostic or screening tool, has not been clinically validated, and must not be presented as one. This report assesses readiness for Phase 2 only; it does not implement modeling or validation.

## Phase 0 completion

- The repository has an explicit data ethics policy prohibiting committed raw video, frames, face crops, and other identifiable child imagery.
- SSBD+ XML and aggregate CSV metadata schemas are supported by offline parsers, including validation of video identifiers, URLs, annotation intervals, and behavior categories.
- A metadata-only access-report path can generate per-video CSV rows from a local manifest, including annotation status and segment counts.
- The access workflow performs no video downloads. Download execution is intentionally absent, and the current data-access report makes no availability claims.

## Phase 1 completion

- `PoseKeypoint` provides a validated numeric representation of frame-level MediaPipe Pose landmarks.
- Numeric keypoints have strict CSV read/write support with a stable schema.
- Fixed-duration, video-scoped windows are generated without crossing video boundaries. Binary labels are derived from overlapping annotations, with `no-class` treated as negative.
- Interpretable numeric features cover wrist and head dominant frequency and periodicity plus inter-wrist distance statistics.
- The single-video pose extraction scaffold keeps decoded frames in memory, atomically writes only numeric keypoint CSV output, and deletes the input video only after successful extraction by default. OpenCV and MediaPipe are optional, lazily loaded runtime dependencies.
- The feature-table pipeline converts numeric keypoint CSV data and local annotations into labeled numeric window features. Its synthetic end-to-end test covers dry-run validation, execution, labeling, feature generation, and CSV round-trip behavior without real media.

## Safety and privacy status

A tracked-file audit found:

- no raw video committed;
- no frames, images, or face crops committed;
- no SSBD XML or CSV data copied into the repository;
- no model artifacts committed.

Only numeric pose keypoints, numeric feature tables, and aggregate metrics are intended for persistence. Raw-video staging and processed-data locations are gitignored, as are common model-artifact formats. Synthetic tests create temporary fixtures outside tracked repository content.

## Verification status

Latest full verification command:

```text
python3 -m pytest
```

Exact result: **53 passed in 2.06s**. Pytest also emits a `pytest-asyncio` `PytestDeprecationWarning` because `asyncio_default_fixture_loop_scope` is unset. This environment-level warning is non-blocking and does not indicate a project test failure.

## Phase 2 readiness and constraints

The repository is ready to add baseline classifiers and child/video-disjoint validation, subject to these mandatory constraints:

- Modeling must consume numeric feature tables only, never raw video, decoded frames, images, or face crops.
- Honest `GroupKFold` validation grouped by child is mandatory. If reliable child identity is unavailable, video-disjoint grouping is the minimum acceptable fallback and must be disclosed.
- The deliberately leaky split ablation must remain isolated under `scripts/diagnostics/`; it must not be exposed as an option on the main baseline runner.
- Results must remain framed as proxy motor-behavior recognition. No diagnostic, autism-screening, clinical-assessment, or clinical-validation claims are permitted.

## Known not-yet-done items

- No real MediaPipe extraction run has been performed.
- Video availability and link attrition have not been verified.
- No baseline model has been implemented or run.
- No validation metrics have been produced.
- No permutation test or leave-one-subject/child-out (LOSO) evaluation has been implemented.

These are Phase 2 or later activities and are not evidence gaps in the tested Phase 1 scaffold itself.
