# MLX ExpertSSD for LivSeek

This repository is the DeepSeek-specific native MLX fork used by LivSeek.

## Baseline

- Official upstream: `https://github.com/ml-explore/mlx.git`
- Upstream commit: `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`
- Initial LivSeek tag: `livseek-preview-v1`

The initial fork state is the exact native source used for the clean
DeepSeek-V4-Flash preview qualification on the 48 GB M5 Pro.

## Native additions

- direct safetensors reads into ExpertSSD slots;
- fixed row-copy and route-planning operations;
- native Markov-LHD state;
- SSD/GPU event coordination;
- fixed-decode MXFP4 expert kernels;
- Python bindings for the LivSeek runtime.

`kg36/mlx-io` remains the independently advancing shared/LivMLX fork. LivSeek
must pin an exact commit from this repository and must not consume a moving
branch from `mlx-io`.

## Build consumer

LivSeek clones a pinned revision into `native/mlx-official-direct` and runs:

```bash
scripts/build_native_mlx.sh
```

Changes shared with LivMLX should be deliberately ported between the forks and
qualified independently on both model families.
