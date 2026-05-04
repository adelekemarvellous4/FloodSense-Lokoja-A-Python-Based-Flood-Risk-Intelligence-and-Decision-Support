# Historical Validation

Historical validation compares modelled flood-prone zones with Sentinel-1 SAR flood evidence.

## SAR Logic

The starter workflow expects a pre-flood raster and a flood/post-flood raster. It applies simple change detection, using a ratio method by default. Pixels with a strong backscatter reduction are treated as candidate observed flood pixels.

## Validation Logic

High and Very High susceptibility classes are treated as predicted flood-prone areas. These are compared against observed SAR-derived flood extent.

Metrics saved to `outputs/validation/validation_metrics.csv`:

- true negatives
- false positives
- false negatives
- true positives
- precision
- recall
- F1 score
- overall accuracy

This is a starter validation method. A stronger version should add permanent-water masking, speckle filtering, terrain shadow masking, and threshold calibration.
