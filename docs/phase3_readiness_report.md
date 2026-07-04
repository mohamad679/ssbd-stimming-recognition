# Phase 3 Readiness Report

## Project Framing

This repository is a proxy motor-behavior recognition project focused on arm-flapping, head-banging, spinning, and no-stereotypy classification from a public research dataset.

It is not an autism diagnosis tool, not a screening tool, and not clinically validated. No clinical, diagnostic, or deployment claim should be made from the current scaffold.

## Phase 3 Scope Summary

Phase 3 adds the statistical and provenance scaffold for the benchmark:

- Leave-one-child-out / leave-one-group-out validation
- Within-group permutation testing
- Artifact provenance manifest tooling
- SVG/XML validation tooling

## What Is Now Possible

The current Phase 3 scaffold makes the following workflows available:

- Run group-disjoint baseline evaluation
- Run LOSO held-out group validation
- Run within-group permutation tests using the same benchmark model configs
- Record checksums for numeric artifacts
- Validate future SVG figures before commit

## What Is Still Not Done

The Phase 3 scaffold does not change the repo into a defended study result. In particular:

- No real SSBD video inference has been run
- No real benchmark metrics are claimed
- No clinical validation has been performed
- No interpretability figures are included yet
- No packaging or model card has been added yet

## Safety And Privacy Guardrails

The repository should continue to enforce these constraints:

- Raw video, extracted frames, images, and model artifacts remain untracked
- Numeric keypoints and derived features are the intended persistence targets
- Generated manifests and reports must be intentionally created and reviewed before commit

## Reproducibility Checklist For A Future Defended Run

1. Resolve or download allowed videos externally
2. Extract numeric keypoints
3. Build the feature table
4. Run group-disjoint baselines
5. Run LOSO
6. Run the permutation test; the final defended run should use 1,000 permutations
7. Build the artifact provenance manifest
8. Validate SVGs if figures are generated

## Readiness Statement

Phase 3 is complete as a scaffold. The project is ready for Phase 4 interpretability skeleton visualizations, while still making no diagnostic, screening, or deployment claim.
