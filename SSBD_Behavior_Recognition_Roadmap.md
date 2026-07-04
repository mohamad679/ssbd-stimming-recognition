# SSBD Self-Stimulatory Behavior Recognition — Project Roadmap

**Purpose:** a small, defensible, video-based behavior-understanding demonstrator connecting to the Idiap PhD position (multimodal gaze/gesture/behavior understanding for healthcare, autism screening applications). Same engineering discipline as the Parkinson wearable project: honest scope, subject(child)-disjoint validation, calibration, permutation testing, no diagnostic claims.

**Non-negotiable framing (put this at the top of the repo README from commit #1):**
> This is a proxy motor-behavior recognition task (arm-flapping / head-banging / spinning vs. no stereotypy) built on a public research dataset. It is **not** an autism diagnostic or screening tool, has not been clinically validated, and must never be described as one.

---

## 0. Data source decision (verified, don't re-search this)

**Primary dataset: SSBD+** — `github.com/sarl-iiitb/ssbdplus-dataset` (Lokegaonkar et al., IEEE Healthcom 2023).
- Extends the original 2013 SSBD with 35 new videos (total ≈60), same XML annotation format, includes an explicit **"no-class"** negative label annotated by clinicians at Bubbles Centre for Autism.
- Ships with a usage doc / parser — check it for the exact XML schema before writing your own loader.
- Chosen as primary because it's the more recently maintained repo (2023 vs. 2013), so video links are more likely to still resolve.

**Fallback / supplementary source: original SSBD** — `rolandgoecke.net/research/datasets/ssbd/` (Rajagopalan, Dhall & Goecke, ICCV 2013 workshop).
- Provides `url-list.pdf` (YouTube URLs), an `Annotations/` folder (XML), and `baseline-results.xlsx` for a literature comparison point.
- Known attrition: multiple papers report only **60 of the original 75 videos** are still downloadable (link rot). Expect this to be worse in 2026 — budget for it, don't be surprised by it.
- Contact on the page: Shyam Rajagopalan, if you need clarification on annotation format.

**Do not use ESBD** (Negin et al.) — multiple papers confirm it is not publicly accessible. Don't burn time chasing it.

**Reference only (not a dependency):** `github.com/antran89/clipping_ssbd_videos` — a toolbox for clipping SSBD videos by XML annotation. Useful to read for annotation-parsing logic even if you don't import it directly.

---

## Repo structure

```
ssbd-stimming-recognition/
  data/
    raw/                  # gitignored — downloaded videos (deleted after keypoint extraction)
    processed/            # gitignored — keypoint time-series only (numeric, no images)
  src/ssbd_behavior/
    acquisition/          # video download + annotation parsing
    pose/                 # MediaPipe keypoint extraction
    features/             # windowing + feature engineering from keypoints
    validation/            # child-disjoint GroupKFold / LOSO + permutation test
    models/                 # baseline classifiers
    evaluation/             # AUROC/AUPRC/Brier/ECE
  scripts/
    diagnostics/            # leaky-split ablation, isolated from day 1 this time
  docs/
  tests/
  configs/
```

---

## Phase 0 — Data Access & Ethics Foundation (do this before writing any model code)

**Tasks:**
1. Clone/read `sarl-iiitb/ssbdplus-dataset`, understand its XML schema and video manifest.
2. Write `scripts/acquisition/resolve_videos.py`: attempt to download every video in the manifest (via `yt-dlp`), log success/failure per video ID. Do the same for the original SSBD `url-list.pdf` as a supplementary pool if SSBD+ coverage is thin.
3. **Immediately after keypoint extraction in Phase 1, delete the raw video file.** Never let more than one video sit on disk decoded at a time if you want to enforce this cleanly — process-then-delete, not download-all-then-process-all.
4. Write `docs/data_ethics_policy.md` up front, stating explicitly: no raw video, no video frames, no face crops are ever committed to git; only numeric pose keypoints and aggregate metrics are persisted; any figure in the repo showing a "case study" uses an anonymized stick-figure skeleton plot, never a real frame.
5. Write `docs/ssbd_data_access_report.md`: table of every video ID attempted, source (SSBD+/original SSBD), download status, and final usable video/segment count. Report this the same way you reported the Daphnet S04/S10 zero-positive-subject issue — as a finding, not a footnote.

**Git commit:** `Add data access report and ethics policy for SSBD+ pipeline`

---

## Phase 1 — Pose Extraction & Windowed Feature Pipeline

**Tasks:**
1. `src/ssbd_behavior/pose/extract_keypoints.py`: run MediaPipe **Pose** (33 keypoints, includes wrists/shoulders/head — sufficient for arm-flapping and head-banging periodicity) frame-by-frame on each downloaded video. Start with Pose, not Holistic — Holistic (adds hand/face mesh) is a stretch upgrade for Phase 4 if wrist-level signal alone proves too weak. This mirrors the "start with the interpretable baseline" approach from the Parkinson project.
2. Persist only the keypoint time-series (`.parquet` or `.csv`, one row per frame per video, columns = keypoint x/y/confidence) to `data/processed/`. Delete the video immediately after.
3. `src/ssbd_behavior/features/windowing.py`: fixed-size sliding windows over the keypoint time-series (e.g. 2–3 seconds), **never crossing a video/child boundary**, aligned against the XML-annotated stimming segments to produce binary window labels (stimming vs. no-class).
4. `src/ssbd_behavior/features/engineering.py`: classical, interpretable features per window — dominant frequency and periodicity strength of wrist/head oscillation (FFT-based, this is the single most informative feature class for arm-flapping/head-banging/spinning), joint-angle angular velocity variance, inter-wrist distance oscillation. This is the same "interpretable-features-first" philosophy as the Daphnet accelerometer baseline — don't jump to a raw-sequence deep model until this baseline is defensible.
5. Tests: unit tests for windowing boundary correctness (no window spans two videos), feature-value sanity checks, and a synthetic-keypoint smoke test (same pattern as `prepare_data.py --synthetic` in the Parkinson repo) so CI doesn't depend on real video access.

**Git commit:** `Implement MediaPipe pose extraction and windowed feature pipeline`

---

## Phase 2 — Baseline Models + Child-Disjoint Validation

**Tasks:**
1. `scripts/run_baselines.py`: class-weighted logistic regression + random forest, binary classification (any stereotypy vs. no-class) at the window level.
2. **Child-disjoint `GroupKFold`, grouped by video/child ID** — this is the exact same leakage risk as Daphnet subject leakage. A child's own gait/motion signature will leak across windows of the same video if you don't group by video ID at minimum, ideally by child if multiple videos exist per child.
3. **Build the leaky-vs-honest ablation script in Phase 2, not as a bolted-on Phase 3 fix.** Last time (Daphnet) this was added only after review feedback — do it proactively this time: `scripts/diagnostics/leaky_split_ablation.py`, isolated, loud warning header, never wired as a CLI flag on the main runner.
4. Metrics: AUROC, AUPRC (expect class imbalance — SSBD+ paper itself reports ~7:1 no-class:stimming ratio), Brier score, ECE.

**Git commit:** `Add child-disjoint baseline classifiers with leaky-split ablation`

---

## Phase 3 — Statistical Rigor (LOSO + Permutation Test, done right the first time)

Lessons from the Daphnet project, applied proactively this time instead of after review:

1. **Leave-One-Child-Out validation**, per-child AUROC/AUPRC/Brier/ECE breakdown table, mean ± SD, with explicit handling (exclude + document, don't silently drop) for any child with zero positive windows.
2. **Permutation test**: shuffle labels **within child**, 1,000 permutations, and — this is the specific bug to avoid — **use the exact same model configuration (same `n_estimators`, same everything) as the main benchmark being defended.** Don't quietly drop to a cheaper model config for speed; if speed is genuinely a problem, say so explicitly in the report rather than silently changing the config.
3. **Provenance manifest from day 1**: `docs/ssbd_artifact_manifest.md` with SHA256 checksums of the processed keypoint dataset used for every downstream report (3-fold benchmark, LOSO, permutation, leaky ablation), so there's never ambiguity later about whether two reports used the same data. Build this in Phase 3, not after someone points out inconsistent numbers.
4. Validate generated SVG figures are well-formed XML before committing (`xml.etree.ElementTree.parse()` as a CI check) — the Daphnet figures had a duplicate-attribute bug that slipped through; add this as an actual CI step this time instead of catching it manually in review.

**Git commit:** `Add LOSO validation, permutation test, and artifact provenance manifest`

---

## Phase 4 — Interpretability (privacy-safe, and directly speaks to "gesture" in the Idiap job description)

1. Feature-importance analysis: which engineered features (wrist periodicity, angular velocity variance, etc.) drive the classifier for each behavior class.
2. **Anonymized skeleton visualization**, not real video frames: for a true positive / false positive / false negative example, plot the MediaPipe keypoints as a stick figure over time (matplotlib), never the underlying video frame or face. This solves the "how do I show a compelling example without publishing a child's face" problem cleanly.
3. Optional stretch: per-behavior-class breakdown (arm-flapping vs. head-banging vs. spinning) instead of only binary any-stereotypy — only attempt this once the binary baseline is solid, same incremental logic as everything else here.

**Git commit:** `Add feature-importance analysis and anonymized skeleton visualizations`

---

## Phase 5 — Packaging & Repo Hygiene

1. `MODEL_CARD.md` — same non-diagnostic, research-use-only framing as the Parkinson project.
2. `docs/limitations.md` — small N (≈60 videos, fewer usable children), YouTube-sourced uncontrolled recording conditions, proxy task (motor stereotypy detection ≠ autism diagnosis), single-modality (pose only, no audio/gaze in this phase).
3. CI workflow: pytest + ruff + the SVG-validity check from Phase 3, same pattern as the Parkinson repo's `.github/workflows`.
4. `docs/interview_talking_points.md` — same "what's implemented / what's not / red-line statements" structure that worked well in the MedShiftLab-CXR repo. Explicitly list what NOT to claim (e.g. "this diagnoses autism," "this matches clinical assessment").
5. One short paragraph in the README connecting this project to the Idiap AIMANT project's stated interest in video-based behavior understanding for healthcare — accurately scoped, not oversold.

**Git commit:** `Add model card, limitations, CI, and interview talking points`

---

## Realistic time estimate

| Phase | Estimate | Main risk |
|---|---|---|
| 0 — Data access & ethics | 0.5–1 day | Video link attrition; budget slack here |
| 1 — Pose + features | 1–2 days | MediaPipe extraction speed over ~60 videos on CPU |
| 2 — Baselines + validation | 0.5 day | Low — reusing established patterns |
| 3 — LOSO + permutation | 0.5 day | Low if done right the first time (see above) |
| 4 — Interpretability | 0.5 day | Low |
| 5 — Packaging | 0.5 day | Low |
| **Total** | **~4–5 days** | Dominated by Phase 0/1 data acquisition |

---

## Kickoff prompt for Claude Code (paste this as your first instruction in VS Code)

> Build the SSBD self-stimulatory behavior recognition project per `SSBD_Behavior_Recognition_Roadmap.md` in this repo, phase by phase, committing after each phase. Non-negotiable rules: never commit raw video, video frames, or face images to git — only numeric pose keypoints and aggregate metrics; enforce child/video-disjoint validation everywhere; build the leaky-split ablation and permutation test proactively in their designated phases, not as an afterthought; use identical model configuration between the main benchmark and its permutation test; validate all generated SVGs as well-formed XML in CI; write an honest data-access/attrition report in Phase 0 before writing any modeling code. Start with Phase 0.