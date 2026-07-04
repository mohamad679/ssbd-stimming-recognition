# Phase 4 Readiness Report

## Project Framing

This repository addresses proxy motor-behavior recognition only. It is not an
autism diagnosis tool, not a screening tool, not clinically validated, and not
deployment-ready.

## Phase 4 Scope Summary

- Phase 4A provides abstract skeleton SVG visualization from numeric keypoints only.
- Phase 4B provides model-native numeric feature-importance and explanation scaffolding.
- No raw frames, videos, child images, or identifying visuals were used.

The explanations summarize native attributes of the existing baseline models.
They are exploratory and non-causal, and they are not diagnostic evidence.

## What Is Now Possible

- Render abstract stick-figure SVGs from numeric keypoints.
- Validate generated SVGs before intentional review and commit.
- Summarize model-native feature importances from the baseline models.
- Generate optional JSON or CSV explanation reports only with explicit `--execute`.

## What Is Still Not Done

- No real SSBD video inference has been run.
- No real benchmark metrics are claimed.
- No clinical validation has been performed.
- No diagnostic or screening use is supported.
- No packaging or model card has been added yet.
- No deployment, API, or frontend has been added.

## Safety And Privacy Guardrails

- Raw videos, frames, images, model artifacts, result artifacts, generated figures, and generated reports remain untracked unless intentionally reviewed.
- Numeric keypoints and derived numeric features are the intended persistence targets.
- Generated SVGs and explanation reports are artifacts and should not be committed casually.
- The Phase 4 workflows do not require raw media or identifying visuals.

## Readiness Statement

Phase 4 is complete as a scaffold. The project is ready for Phase 5 packaging,
model card, limitations, and CI work while still making no diagnostic, clinical,
deployment, or state-of-the-art claim.
