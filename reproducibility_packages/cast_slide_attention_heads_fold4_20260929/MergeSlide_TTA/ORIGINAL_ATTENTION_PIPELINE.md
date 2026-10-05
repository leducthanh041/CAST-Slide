# Original TITAN Pool-Attention Visualization

This is the full pipeline that produces the individual original-jet maps and
the six-task continual-learning figure.

## Inputs

- Fixed six-WSI manifest:
  `logs/attention_maps/fold4_train_dominant_specimen_jet/selected_slides.json`
- Raw SVS paths and feature H5 paths recorded in that manifest.
- OOD fold-4 checkpoints under `checkpoints_ood/`.
- TITAN task prompts in `task_prompts.pt`.

## Attention extraction

1. `create_continual_attention_grid.py::stage_checkpoint` selects:
   stage 1 `task_0.pt`, stages 2-5 `merged_task_1.pth` through
   `merged_task_4.pth`, and stage 6 `merged_final.pth`.
2. `create_cast_slide_heatmaps.py::make_tta_model` loads TITAN and constructs
   naive CAST-Slide with the adapted student used for inference.
3. For each stage, one fresh prefix model is carried causally through the six
   configured WSIs with `episodic=False`. Before each update, optional
   `pre_wsi_scores.h5` attention is captured. Then
   `create_continual_attention_grid.py::naive_prediction` chooses the
   deterministic spatial K=400 adaptation bag without resetting the model.
4. Immediately after adapting the current WSI, full-patch post-WSI attention
   is captured and becomes the attention shown in the main figure.
5. `create_cast_slide_heatmaps.py::aggregate_scores` creates spatial K=400
   context bags covering every patch three times.
6. `create_cast_slide_heatmaps.py::score_attention_context_bag` installs
   `cast_slide/titan_attention.py::TitanAttentionCapture` and captures native
   TITAN contrastive pooling query-to-patch attention from the adapted student.
7. Scores for repeated contexts are aggregated by the median. The result is
   `titan_pool_attention_median` for every patch in the WSI.

## Saved score artifacts

For each task and stage, `create_continual_attention_grid.py` creates:

- `scores.h5`: coordinates, K=400 adaptation indices, raw/median/percentile
  pooling-attention scores.
- `metadata.json`: checkpoint, prediction, inference and attention metadata.
- `*_titan_pool_attention_blockmap.h5`: CLAM-compatible block map.

`reference_percentiles` uses one shared reference made from all six stages of
the same WSI. This makes stage-to-stage colors comparable within a WSI.

## Original rendering

`create_cast_slide_heatmaps.py::render_maps` renders exactly:

```text
titan_pool_attention_percentile -> jet -> titan_pool_attention.jpg
```

CLAM supplies tissue segmentation, coordinate-aware patch overlay and blending
with the raw SVS. Each result is cropped to the fixed specimen region and
placed on a 2048 x 2048 white canvas without stretching.

## Figure composition

`create_continual_attention_grid.py::compose_grid` creates:

- six dataset rows;
- original WSI in column 1;
- continual stages 1-6 in columns 2-7;
- vertical stage separators;
- horizontal continual-learning arrow;
- vertical jet Attention Score colorbar.

Final output:

`logs/attention_maps/fold4_train_dominant_specimen_jet/region_level_attention_maps_6tasks_fold4_naive_restored.jpg`

## Important protocol note

The current figure uses an online causal stream of the six explicitly selected
WSIs. It is a focused diagnostic run, not adaptation over the complete task
test split. Each stage starts independently from its own source prefix.
