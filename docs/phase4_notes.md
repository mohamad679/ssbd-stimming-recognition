# Phase 4 notes

Phase 4A adds an abstract skeleton SVG visualization scaffold derived from
numeric pose keypoints only.

Phase 4B adds model-native numeric feature-importance and explanation
scaffolding for the existing logistic-regression and random-forest baselines.
These importances are exploratory and non-causal. They are not diagnostic or
screening evidence.

The visualization path does not use raw frames, child images, videos, or other
identifying visuals. It renders only abstract stick-figure lines and joints from
numeric coordinates.

The feature-importance path uses synthetic or intentionally supplied numeric
feature tables only. It does not read raw frames, images, or videos. Optional
JSON or CSV explanation reports are written only when `--execute` is provided.

Generated SVGs and explanation reports are output artifacts. They should not be
committed unless they have been intentionally reviewed and are being added on
purpose.

This interpretability scaffold remains non-diagnostic, research-scoped, and not
clinically validated.

Phase 4 is complete as a scaffold. Phase 5 packaging, model card, limitations,
and CI work is next; none of that Phase 5 implementation is included here.
