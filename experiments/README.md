# Experiments

These are optional prior-run scripts. The main demo entry point is still:

```bash
python3 chat_steer.py
```

The scripts in this folder are useful when you want to reproduce or extend
specific comparisons. They write local outputs to root-level `results/`,
`captures/`, or `data/` directories, which are ignored by git.

Example:

```bash
python3 experiments/experiment_dim_n12.py
```
