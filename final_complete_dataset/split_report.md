# Dataset Split Report

## Overview

- **Total dataset span**: 21 days from 20250519 to 20250608
- **Total files**: 181440

## Sets Summary

| Set | Days | Files | QoE Min | QoE Max | QoE Mean | QoE Median |
|-----|------|-------|---------|---------|----------|------------|
| Train | 14 | 120960 | 60.08 | 99.99 | 87.87 | 88.97 |
| Validation | 3 | 25920 | 60.08 | 99.99 | 88.71 | 90.00 |
| Test | 4 | 34560 | 60.08 | 99.99 | 87.27 | 88.28 |

## QoE Distribution

| Set | Very Low<br>(0-20) | Low<br>(20-40) | Medium-Low<br>(40-60) | Medium<br>(60-80) | Medium-High<br>(80-90) | High<br>(90-100) |
|-----|-------------------|----------------|----------------------|------------------|------------------------|-----------------|
| Train | 0.0% | 0.0% | 0.0% | 19.0% | 35.7% | 45.2% |
| Validation | 0.0% | 0.0% | 0.0% | 16.7% | 33.3% | 50.0% |
| Test | 0.0% | 0.0% | 0.0% | 20.8% | 37.5% | 41.7% |

## Day Allocation

### Training Set

| Day | Files | QoE Min | QoE Max | QoE Mean |
|-----|-------|---------|---------|----------|
| 20250519 | 8640 | 60.08 | 99.99 | 88.50 |
| 20250520 | 8640 | 60.08 | 99.99 | 88.71 |
| 20250521 | 8640 | 60.08 | 99.99 | 88.76 |
| 20250522 | 8640 | 60.08 | 99.99 | 88.88 |
| 20250523 | 8640 | 60.08 | 99.99 | 88.67 |
| 20250524 | 8640 | 60.08 | 99.99 | 84.64 |
| 20250525 | 8640 | 60.08 | 99.99 | 86.79 |
| 20250526 | 8640 | 60.08 | 99.99 | 88.79 |
| 20250527 | 8640 | 60.08 | 99.99 | 88.78 |
| 20250528 | 8640 | 60.08 | 99.99 | 88.81 |
| 20250529 | 8640 | 60.08 | 99.99 | 88.60 |
| 20250530 | 8640 | 60.08 | 99.99 | 88.72 |
| 20250531 | 8640 | 60.08 | 99.99 | 84.70 |
| 20250601 | 8640 | 60.08 | 99.99 | 86.88 |

### Validation Set

| Day | Files | QoE Min | QoE Max | QoE Mean |
|-----|-------|---------|---------|----------|
| 20250602 | 8640 | 60.08 | 99.99 | 88.84 |
| 20250603 | 8640 | 60.08 | 99.99 | 88.61 |
| 20250604 | 8640 | 60.08 | 99.99 | 88.67 |

### Testing Set

| Day | Files | QoE Min | QoE Max | QoE Mean |
|-----|-------|---------|---------|----------|
| 20250605 | 8640 | 60.08 | 99.99 | 88.82 |
| 20250606 | 8640 | 60.08 | 99.99 | 88.81 |
| 20250607 | 8640 | 60.08 | 99.99 | 84.77 |
| 20250608 | 8640 | 60.08 | 99.99 | 86.68 |

## Visualizations

Visualizations of the QoE distributions are available in the 'visualizations' directory:

- QoE distribution comparison: [qoe_distribution_comparison.png](visualizations/qoe_distribution_comparison.png)
- QoE violin plot comparison: [qoe_violin_comparison.png](visualizations/qoe_violin_comparison.png)
- QoE by day: [qoe_by_day.png](visualizations/qoe_by_day.png)
