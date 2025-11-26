# Running Complete Visualizations for VideoMAE Training

After your VideoMAE training completes on the HPC cluster, follow these steps to generate all visualizations including Grad-CAM, embeddings, and error galleries.

## CLI-Only Usage (Recommended)

You can now run all visualizations purely via CLI commands without creating any python context files.

```bash
conda activate deep_impact_env

python -m visualizations.run_all \
  --run-dir /scratch/st-lyndiacw-1/gyan/vmae_result/vmae_full_base \
  --save-dir /scratch/st-lyndiacw-1/gyan/vmae_result/vmae_full_base/visualizations \
  --val-csv /scratch/st-lyndiacw-1/gyan/train_dataset/val.csv \
  --enable-gradcam \
  --enable-embedding \
  --enable-galleries \
  --embedding-method umap \
  --no-show
```

## Explanation of Arguments

- `--run-dir`: Directory containing your training outputs (metrics, predictions, config)
- `--save-dir`: Where to save all visualization plots
- `--val-csv`: Path to your validation CSV file (required for advanced visualizations)
- `--enable-gradcam`: Generate Grad-CAM attention maps for sample videos
- `--enable-embedding`: Generate 2D projection of learned representations (UMAP or t-SNE)
- `--enable-galleries`: Generate galleries of correct predictions and errors
- `--embedding-method`: Choose `umap` (faster, recommended) or `tsne`
- `--no-show`: Don't display plots interactively (important for HPC)
- `--max-samples`: Limit number of samples to load for galleries/Grad-CAM (default: 50)

## Generated Visualizations

After running, you'll have these visualizations in the `visualizations/` directory:

### Standard Plots (Always Generated)
1. **loss.png** - Training and validation loss curves over epochs
2. **f1.png** - Training and validation F1 score curves
3. **roc.png** - ROC curve with AUC score
4. **precision_recall.png** - Precision-Recall curve
5. **metrics_vs_threshold.png** - How metrics change with classification threshold
6. **probability_distributions.png** - Distribution of predicted probabilities
7. **confusion_matrix.png** - Confusion matrix at 0.5 threshold

### Grad-CAM Visualizations (with --enable-gradcam)
8. **gradcam_sample_0.png**, **gradcam_sample_1.png**, etc. - Attention heatmaps overlaid on video frames showing what the model focuses on

### Embedding Visualization (with --enable-embedding)
9. **embedding.png** - 2D UMAP/t-SNE projection of video representations, colored by true labels and shaped by predictions

### Qualitative Galleries (with --enable-galleries)
10. **error_gallery.png** - Grid of misclassified examples with predictions
11. **correct_gallery.png** - Grid of correctly classified examples

## Expected Runtime

- **Standard plots**: ~5-10 seconds
- **Grad-CAM**: ~30-60 seconds (processes subset of videos)
- **Embeddings**: ~2-5 minutes (depends on validation set size)
- **Galleries**: ~30-60 seconds

Total: ~3-7 minutes for all visualizations

## Troubleshooting

### If Grad-CAM fails
The script tries to identify the last transformer block automatically. If it fails, you'll see a warning but other visualizations will still generate.

### If embeddings are slow
- Use `--embedding-method umap` instead of `tsne` (UMAP is faster)
- The script processes your entire validation set to extract features

### Out of memory errors
If you get CUDA OOM during visualization:
- The script defaults to batch size 2 for VideoMAE.
- Or run on CPU by setting `CUDA_VISIBLE_DEVICES=""` before the command.
