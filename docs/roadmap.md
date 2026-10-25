# Learning and experiment roadmap

## Completed

**01 — One-step dynamics prediction**: 4-to-3 state-delta MLP; normalized training; test-set original-scale one-step MSE; no-change baseline. Original observations recorded in `results/first_run_observed.json`.

## Next

**02 — Multi-step rollout evaluation**: hold initial true state fixed, supply the same action sequence to simulator and learned model, predict 1/5/10/20/50/100 steps, plot per-step and per-dimension errors. Distinguish open-loop from one-step teacher-forced estimates.

**03 — Model Predictive Control**: random shooting/CEM planning for torque sequences, execute only first torque and replan. Compare with random policy and stronger model-based baseline across multiple seeds.

**04 — Improve model quality**: circularity constraints, multi-step training loss, ensembles for epistemic uncertainty, coverage shifts and out-of-distribution evaluation.

**05 — Visual world model**: small image encoder and latent dynamics, preferably with Colab GPU for expensive experiments.

**06 — Mini-VLA**: camera observations + short text instructions + simple discrete/continuous action output before using heavier pretrained models.

## Good experiment habits

For every experiment, commit code/config changes and record: goal, dataset/environment, seeds, hyperparameters, train/validation/test metrics on comparable scales, baseline, plots, surprising failures, and next question. Use separate run directories. Never commit credentials or large checkpoints accidentally.
