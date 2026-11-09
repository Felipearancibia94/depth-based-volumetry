---
name: lab-eval
description: Compare the synthetic lab evaluation before and after a change to event rules, the tracker, spatial estimation or the pipeline. Use when editing src/perception_lab/events, vision/tracker.py, spatial or apps/pets.
allowed-tools: Bash(uv run lab eval*), Bash(uv run lab run*), Bash(uv run pytest*)
---

# Lab evaluation

1. Before editing, run `uv run lab eval` and keep the metrics (per-activity precision/recall,
   false events, id switches).
2. Make the change.
3. Run `uv run lab eval` again with the same options, and with controlled noise if the change
   affects robustness (`detector.gt_miss_rate`, `detector.gt_jitter_px` in the config).
4. Report both runs side by side and what moved. Note that the synthetic cat rests for about a
   minute, so `resting_or_sleeping` needs `rules.stationary_s = 30` to fire.
5. State plainly that these numbers check the plumbing on a synthetic scene; they are not
   real-world accuracy. Never put them in docs as hardware or model results.
6. Run `/verify`.
