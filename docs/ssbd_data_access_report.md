# SSBD Data Access Report

## Status

This is the Phase 0 reporting template. No videos have been downloaded by the repository scaffold, and no availability claims have yet been verified.

## Dataset source summary

### Primary source: SSBD+

- Repository: `sarl-iiitb/ssbdplus-dataset`
- Role: primary source and manifest for this project.
- Roadmap summary: extends the original SSBD collection and includes an explicit `no-class` negative annotation.
- Pending Phase 0 work: record each manifest entry, annotation availability, access attempt, and usable annotated segment count.

### Fallback source: original SSBD

- Dataset page: `rolandgoecke.net/research/datasets/ssbd/`
- Role: supplementary fallback if usable SSBD+ coverage is insufficient.
- Manifest reference: `url-list.pdf`, with XML annotations provided separately by the dataset source.
- Known risk from the roadmap: historical link rot; current availability must be measured and reported rather than assumed.

## Access-attempt table

One row must be recorded for every attempted video. `URL or manifest reference` may contain a public URL or a stable reference into a locally obtained dataset manifest; it must not point to committed raw media.

| Attempted video ID | Source | URL or manifest reference | Download status | Failure reason | Annotation status | Usable segment count | Notes |
|---|---|---|---|---|---|---:|---|
| _Not attempted_ | — | — | not_attempted | — | unknown | 0 | Template row; replace with observed entries. |

## Summary counts

| Measure | Count |
|---|---:|
| Manifest entries reviewed | 0 |
| Videos attempted | 0 |
| Videos accessible | 0 |
| Videos unavailable | 0 |
| Videos with usable annotations | 0 |
| Total usable annotated segments | 0 |

## Attrition and link-rot findings

Record observed attrition by source and reason, including removed or private videos, unavailable hosts, malformed references, access restrictions, missing annotations, and annotations with no usable segments. Distinguish observed failures from entries that have not yet been attempted.

No access attempts have been run in this scaffold, so there are no measured attrition findings yet.

## Reporting rule

Missing or unavailable videos are reported as findings, not silently ignored. Every manifest entry considered for the study must remain visible in the access table with its status and, where applicable, a failure reason.

## Ethics and storage note

The repository stores no raw video, frames, face crops, or identifiable child imagery. See [Data Ethics Policy](data_ethics_policy.md). Only numeric pose keypoints and aggregate metrics may be persisted in later phases.
