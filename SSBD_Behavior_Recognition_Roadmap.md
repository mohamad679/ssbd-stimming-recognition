# SSBD Behavior Recognition Roadmap

This document summarizes the public-facing project roadmap and keeps only durable engineering information. Internal implementation prompts, commit instructions, and time estimates are intentionally excluded.

## Project scope

The repository focuses on a privacy-conscious motor-behavior recognition research workflow built around numeric pose keypoints, engineered temporal features, group-disjoint validation, and conservative reporting.

The project is research-only. It is not deployment-ready and should not be presented as a clinical or screening system.

## Roadmap summary

1. Data access and ethics guardrails
   - Keep raw media and frame-level imagery out of version control.
   - Persist only numeric keypoints, derived features, reports, manifests, and safe abstract visualizations.
   - Document source access, unavailable videos, and final usable cohort size.

2. Pose extraction and feature generation
   - Extract numeric pose keypoints from permitted local video inputs.
   - Delete temporary raw media after successful keypoint extraction.
   - Build fixed-duration windows and interpretable temporal features.

3. Baselines and validation
   - Evaluate class-weighted logistic regression and random forest baselines.
   - Use group-disjoint validation to avoid leakage between train and test windows.
   - Maintain a separate leaky-split diagnostic only as an explicit ablation.

4. Statistical rigor
   - Run leave-one-group-out validation where possible.
   - Run within-group permutation testing with the same defended model configuration.
   - Write provenance manifests with hashes for generated numeric artifacts.

5. Interpretability
   - Report model-native feature importance.
   - Use abstract skeleton SVGs only; do not publish real frames or identifying imagery.

6. Packaging and documentation
   - Keep model-card, limitations, data-ethics, benchmark-report, and final-status documentation aligned.
   - Preserve conservative claims and explicit non-deployment boundaries.

## Current public status

The repository now contains a completed accessible-video benchmark report, model card, limitations documentation, privacy-safe artifact packaging helpers, and synthetic tests for the core numeric pipeline.

## Remaining engineering priorities

- Harden benchmark subprocess execution with timeouts.
- Verify remote model artifacts by hash before use.
- Keep dependency audit tooling active in CI.
- Continue reducing duplication between CLI scripts and package modules.
- Add stronger coverage and static-analysis checks as the project matures.
