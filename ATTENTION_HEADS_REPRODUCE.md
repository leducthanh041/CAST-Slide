# Fold-4 TITAN Attention-Head Reproduction Bundle

This bundle contains the code and configuration used to generate the fold-4
continual attention maps for TITAN tensor head indices 4, 10, and 11.

## Expected directory layout

Extract the archive so that `MergeSlide_TTA` and `CLAM` are siblings:

```text
WSI/
  MergeSlide_TTA/
  CLAM/
```

The bundle intentionally excludes raw SVS slides, preprocessed H5 features,
OOD checkpoints, generated heatmaps, and other large experiment artifacts.
Provide these separately using the same relative layout or override paths with:

```bash
MERGESLIDE_DATA_ROOT=/path/to/dataset
MERGESLIDE_CHECKPOINT_ROOT=/path/to/checkpoints_ood
MERGESLIDE_RAW_ROOT_PRIMARY=/path/to/data_raw_BRCA_LUSC_RCC
MERGESLIDE_RAW_ROOT_AUXILIARY=/path/to/data_raw_CESC_ESCA_TGCT
```

The selected slide IDs and the original path metadata are included under
`reproduction_manifests/`. Paths in those JSON files may need to be remapped
on the destination server.

## Run Head 4

From `MergeSlide_TTA`:

```bash
sbatch scripts/slurm_head4_continual_attention_grid.sh
```

## Run supplementary Heads 10 and 11

```bash
sbatch --job-name=attn_head10_supp \
  --export=ALL,HEAD_INDEX=10 \
  scripts/slurm_supplementary_attention_head.sh

sbatch --job-name=attn_head11_supp \
  --export=ALL,HEAD_INDEX=11 \
  scripts/slurm_supplementary_attention_head.sh
```

The Slurm launchers contain cluster-specific environment and helper paths.
Update their conda, Hugging Face cache, GPU reservation, and log paths when
moving to another cluster. The portable shell launchers are:

```text
scripts/run_head4_continual_attention_grid.sh
scripts/run_supplementary_attention_head.sh
```

## Core implementation

- `create_continual_attention_grid.py`: orchestration, TTA, score extraction,
  per-stage rendering, and final grid composition.
- `create_cast_slide_heatmaps.py`: CAST-Slide model construction and CLAM
  rendering helpers.
- `cast_slide/titan_attention.py`: native TITAN attention hooks.
- `configs/attention_maps/ood_fold4_naive_6task_head4.yaml`: fold-4 runtime,
  TTA, slide-selection, and visualization configuration.

