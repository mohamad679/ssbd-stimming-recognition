# Scripts

`run_distilled_ms_stf.py` runs Stage F D-MS-STF ablations on an existing
privacy-safe numeric feature CSV. Multi-scale columns must use the `ms_` prefix
unless `--multiscale-prefix` is supplied. The command requires an explicit
output directory and writes only numeric CSV/JSON reports. See
`docs/stage_f_distilled_ms_stf.md` for the leakage-safe workflow.

`benchmark/run_stage_g_d_ms_stf_colab.py` is the numeric-only Colab orchestration
entry point for existing SSBD+ keypoint and feature artifacts. It adds missing
`ms_*` columns, runs the Stage F script with GroupKFold and LOSO, supports 2/5
permutation smoke runs and 1,000-permutation final runs, and optionally creates
an allowlisted safe result zip. See `docs/stage_g_d_ms_stf_colab.md`.

All data-bearing commands require explicit local paths. Generated datasets and
result archives stay out of version control.
