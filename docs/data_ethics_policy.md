# Data Ethics Policy

## Scope and claims

This repository is research-only and has not been clinically validated. It is a proxy motor-behavior recognition project and is not an autism diagnostic or screening tool. Its outputs must never be presented as a diagnosis, screening result, clinical assessment, or substitute for professional judgment.

## Prohibited repository content

No raw video, video frames, face crops, or other identifiable child imagery may be committed to this repository. This prohibition applies to source material, intermediate files, debugging output, test fixtures, reports, and figures.

Only numeric pose keypoints and aggregate metrics may be persisted. Numeric keypoints must not be accompanied by source frames or imagery that could identify a child.

## Privacy-safe visualizations

Any visual case study must use anonymized stick-figure skeleton plots only. Real frames, face images, thumbnails, and identifiable backgrounds must not appear in repository artifacts or documentation.

## Raw-video lifecycle

In later phases, each raw video must be deleted immediately after its numeric keypoints have been extracted. Processing must follow a process-then-delete workflow so raw videos do not accumulate on disk. Raw-video locations must remain outside git-tracked paths.

## Reporting

Unavailable videos, link rot, annotation gaps, and excluded material must be reported as study findings. They must not be silently omitted from access or attrition reports.
