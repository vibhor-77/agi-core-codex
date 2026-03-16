# Split Policy

`agi-core-codex` treats splits as first-class experiment inputs.

- `train-dev`: rapid iteration, ablations, debugging
- `train-val`: model selection and checkpoint gating
- `public-eval`: rare checkpoint-only scoring, never tuning

The repository ships with tiny fixture splits for tests and smoke runs. Real benchmark runs should create explicit split files and keep them under version control or artifact tracking.

