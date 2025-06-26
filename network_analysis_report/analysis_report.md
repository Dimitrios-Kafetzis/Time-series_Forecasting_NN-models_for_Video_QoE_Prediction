# Network Data Analysis Report

## Dataset Overview

- Total files: 2185
- Total measurements: 10925
- Time range: 2025-04-02 12:19:46 to 2025-05-10 16:56:40
- Time span: 38 days

## Significant Time Gaps

| Start Time | End Time | Duration (days) | Duration (hours) |
|------------|----------|-----------------|------------------|
| 2025-04-02 13:50:38 | 2025-04-08 12:34:52 | 5.9 | 142.7 |
| 2025-04-08 13:35:13 | 2025-04-08 15:33:54 | 0.1 | 2.0 |
| 2025-04-08 16:03:56 | 2025-05-09 10:00:11 | 30.7 | 737.9 |
| 2025-05-09 14:34:28 | 2025-05-10 12:46:44 | 0.9 | 22.2 |

## Parameter Statistics

| Parameter | Count | Mean | Median | Std Dev | Min | Max | Q1 | Q3 |
|-----------|-------|------|--------|---------|-----|-----|----|----|
| throughput | 10925 | 1179.40 | 1154.50 | 225.75 | 482.30 | 2070.90 | 1087.40 | 1344.00 |
| packets_lost | 10925 | 0.75 | 0.00 | 1.08 | 0.00 | 14.00 | 0.00 | 1.00 |
| packet_loss_rate | 10925 | 0.32 | 0.00 | 0.46 | 0.00 | 4.60 | 0.00 | 0.40 |
| jitter | 10925 | 20.60 | 20.82 | 2.19 | 10.91 | 27.63 | 19.87 | 21.70 |
| qoe | 10925 | 82.47 | 91.53 | 20.67 | 0.03 | 99.99 | 72.28 | 97.76 |

### Distribution Analysis

| Parameter | Skewness | Kurtosis | Normal Distribution? |
|-----------|----------|----------|----------------------|
| throughput | 0.09 | 2.35 | No |
| packets_lost | 2.00 | 6.41 | No |
| packet_loss_rate | 1.92 | 4.88 | No |
| jitter | -1.31 | 3.66 | No |
| qoe | -1.44 | 1.54 | No |

## Parameter Correlations

| Parameter | Throughput | Packets Lost | Packet Loss Rate | Jitter | QoE |
|-----------|------------|--------------|------------------|--------|-----|
| throughput | 1.00 | 0.15 | 0.02 | -0.63 | -0.05 |
| packets_lost | 0.15 | 1.00 | 0.94 | -0.03 | -0.37 |
| packet_loss_rate | 0.02 | 0.94 | 1.00 | 0.05 | -0.36 |
| jitter | -0.63 | -0.03 | 0.05 | 1.00 | 0.02 |
| qoe | -0.05 | -0.37 | -0.36 | 0.02 | 1.00 |

## QoE Relationships

### Correlations with QoE

| Parameter | Correlation with QoE |
|-----------|----------------------|
| throughput | -0.05 |
| packets_lost | -0.37 |
| packet_loss_rate | -0.36 |
| jitter | 0.02 |

### Linear Regression Models

| Parameter | Coefficient | Intercept | R² |
|-----------|-------------|-----------|----|
| throughput | -0.0044 | 87.7079 | 0.0024 |
| packets_lost | -7.0911 | 87.7662 | 0.1373 |
| packet_loss_rate | -15.9904 | 87.5379 | 0.1290 |
| jitter | 0.1727 | 78.9082 | 0.0003 |

### Multiple Regression Model

**R²:** 0.1383

**Coefficients:**

| Parameter | Coefficient |
|-----------|-------------|
| throughput | 0.0010 |
| packets_lost | -5.5108 |
| packet_loss_rate | -3.9385 |
| jitter | 0.1973 |

**Intercept:** 82.6339

**QoE Prediction Formula:**

`QoE = 0.0010 × throughput + -5.5108 × packets_lost + -3.9385 × packet_loss_rate + 0.1973 × jitter + 82.6339`

## Temporal Patterns

### Hourly Patterns

Average values by hour of day:

**Throughput:**

| Hour | Mean Value | Sample Count |
|------|------------|---------------|
| 10:00 | 1202.67 | 915 |
| 11:00 | 1215.79 | 930 |
| 12:00 | 1189.08 | 2170 |
| 13:00 | 1179.56 | 3220 |
| 14:00 | 1192.59 | 1445 |
| 15:00 | 1141.85 | 1325 |
| 16:00 | 1129.49 | 920 |

**Packet_loss_rate:**

| Hour | Mean Value | Sample Count |
|------|------------|---------------|
| 10:00 | 0.29 | 915 |
| 11:00 | 0.26 | 930 |
| 12:00 | 0.19 | 2170 |
| 13:00 | 0.30 | 3220 |
| 14:00 | 0.32 | 1445 |
| 15:00 | 0.39 | 1325 |
| 16:00 | 0.66 | 920 |

**Jitter:**

| Hour | Mean Value | Sample Count |
|------|------------|---------------|
| 10:00 | 20.03 | 915 |
| 11:00 | 20.19 | 930 |
| 12:00 | 19.76 | 2170 |
| 13:00 | 20.35 | 3220 |
| 14:00 | 21.06 | 1445 |
| 15:00 | 21.59 | 1325 |
| 16:00 | 22.23 | 920 |

**Qoe:**

| Hour | Mean Value | Sample Count |
|------|------------|---------------|
| 10:00 | 91.16 | 915 |
| 11:00 | 84.39 | 930 |
| 12:00 | 84.65 | 2170 |
| 13:00 | 83.44 | 3220 |
| 14:00 | 83.90 | 1445 |
| 15:00 | 79.09 | 1325 |
| 16:00 | 65.93 | 920 |

### Daily Patterns

Average values by day of week:

**Throughput:**

| Day | Mean Value | Sample Count |
|-----|------------|---------------|
| Tuesday | 1093.54 | 1425 |
| Wednesday | 1207.44 | 1425 |
| Friday | 1189.17 | 4270 |
| Saturday | 1190.11 | 3805 |

**Packet_loss_rate:**

| Day | Mean Value | Sample Count |
|-----|------------|---------------|
| Tuesday | 0.00 | 1425 |
| Wednesday | 0.26 | 1425 |
| Friday | 0.25 | 4270 |
| Saturday | 0.53 | 3805 |

**Jitter:**

| Day | Mean Value | Sample Count |
|-----|------------|---------------|
| Tuesday | 20.89 | 1425 |
| Wednesday | 18.53 | 1425 |
| Friday | 20.79 | 4270 |
| Saturday | 21.04 | 3805 |

**Qoe:**

| Day | Mean Value | Sample Count |
|-----|------------|---------------|
| Tuesday | 99.23 | 1425 |
| Wednesday | 70.36 | 1425 |
| Friday | 86.69 | 4270 |
| Saturday | 75.98 | 3805 |

### Time of Day Patterns

Average values by time of day category:

**Throughput:**

| Time of Day | Mean Value | Sample Count |
|-------------|------------|---------------|
| Morning | 1209.28 | 1845 |
| Noon | 1183.39 | 5390 |
| Afternoon | 1158.64 | 3690 |

**Packet_loss_rate:**

| Time of Day | Mean Value | Sample Count |
|-------------|------------|---------------|
| Morning | 0.27 | 1845 |
| Noon | 0.26 | 5390 |
| Afternoon | 0.43 | 3690 |

**Jitter:**

| Time of Day | Mean Value | Sample Count |
|-------------|------------|---------------|
| Morning | 20.11 | 1845 |
| Noon | 20.11 | 5390 |
| Afternoon | 21.54 | 3690 |

**Qoe:**

| Time of Day | Mean Value | Sample Count |
|-------------|------------|---------------|
| Morning | 87.75 | 1845 |
| Noon | 83.93 | 5390 |
| Afternoon | 77.69 | 3690 |

## Anomalies

Detected 941 anomalies using Z-score threshold of 3.0

| Timestamp | Parameter | Value | Z-Score |
|-----------|-----------|-------|---------|
| 2025-04-02 12:20:23 | packet_loss_rate | 2.30 | 4.27 |
| 2025-04-02 12:20:36 | packets_lost | 4.00 | 3.01 |
| 2025-04-02 12:21:15 | packets_lost | 4.00 | 3.01 |
| 2025-04-02 12:26:24 | qoe | 3.77 | -3.81 |
| 2025-04-02 12:26:26 | qoe | 3.77 | -3.81 |
| 2025-04-02 12:26:28 | qoe | 3.77 | -3.81 |
| 2025-04-02 12:26:30 | qoe | 3.77 | -3.81 |
| 2025-04-02 12:26:32 | qoe | 3.77 | -3.81 |
| 2025-04-02 12:27:13 | qoe | 2.78 | -3.85 |
| 2025-04-02 12:27:15 | qoe | 2.78 | -3.85 |
| 2025-04-02 12:27:17 | qoe | 2.78 | -3.85 |
| 2025-04-02 12:27:19 | qoe | 2.78 | -3.85 |
| 2025-04-02 12:27:21 | qoe | 2.78 | -3.85 |
| 2025-04-02 12:27:30 | packets_lost | 5.00 | 3.94 |
| 2025-04-02 12:27:30 | qoe | 12.19 | -3.40 |
| 2025-04-02 12:27:32 | qoe | 12.19 | -3.40 |
| 2025-04-02 12:27:34 | qoe | 12.19 | -3.40 |
| 2025-04-02 12:27:36 | packets_lost | 4.00 | 3.01 |
| 2025-04-02 12:27:36 | qoe | 12.19 | -3.40 |
| 2025-04-02 12:27:38 | qoe | 12.19 | -3.40 |
| 2025-04-02 12:28:03 | qoe | 15.32 | -3.25 |
| 2025-04-02 12:28:05 | qoe | 15.32 | -3.25 |
| 2025-04-02 12:28:07 | qoe | 15.32 | -3.25 |
| 2025-04-02 12:28:09 | qoe | 15.32 | -3.25 |
| 2025-04-02 12:28:11 | qoe | 15.32 | -3.25 |
| 2025-04-02 12:31:07 | packet_loss_rate | 2.30 | 4.27 |
| 2025-04-02 12:34:30 | packet_loss_rate | 2.30 | 4.27 |
| 2025-04-02 12:35:46 | packets_lost | 4.00 | 3.01 |
| 2025-04-02 12:36:20 | qoe | 5.45 | -3.72 |
| 2025-04-02 12:36:22 | qoe | 5.45 | -3.72 |
| 2025-04-02 12:36:24 | qoe | 5.45 | -3.72 |
| 2025-04-02 12:36:26 | qoe | 5.45 | -3.72 |
| 2025-04-02 12:36:28 | qoe | 5.45 | -3.72 |
| 2025-04-02 12:37:50 | packet_loss_rate | 2.30 | 4.27 |
| 2025-04-02 12:42:40 | qoe | 8.14 | -3.59 |
| 2025-04-02 12:42:42 | qoe | 8.14 | -3.59 |
| 2025-04-02 12:42:44 | qoe | 8.14 | -3.59 |
| 2025-04-02 12:42:46 | qoe | 8.14 | -3.59 |
| 2025-04-02 12:42:48 | qoe | 8.14 | -3.59 |
| 2025-04-02 12:45:34 | packets_lost | 4.00 | 3.01 |
| 2025-04-02 12:50:07 | qoe | 17.36 | -3.15 |
| 2025-04-02 12:50:09 | qoe | 17.36 | -3.15 |
| 2025-04-02 12:50:11 | jitter | 12.59 | -3.66 |
| 2025-04-02 12:50:11 | qoe | 17.36 | -3.15 |
| 2025-04-02 12:50:13 | qoe | 17.36 | -3.15 |
| 2025-04-02 12:50:15 | qoe | 17.36 | -3.15 |
| 2025-04-02 12:50:28 | jitter | 12.50 | -3.70 |
| 2025-04-02 12:50:40 | jitter | 12.72 | -3.60 |
| 2025-04-02 12:50:44 | packets_lost | 4.00 | 3.01 |
| 2025-04-02 12:50:46 | packet_loss_rate | 2.20 | 4.05 |

*Showing only the first 50 of 941 anomalies.*

## Recommendations for Synthetic Data Generation

Based on the analysis, consider the following when generating synthetic data:

### 1. Parameter Distributions

- Throughput: Not normally distributed. Consider a custom distribution or transformation.
- Packets_lost: Not normally distributed. Consider a custom distribution or transformation.
  - Positively skewed (right-tailed). Consider log-normal or gamma distribution.
- Packet_loss_rate: Not normally distributed. Consider a custom distribution or transformation.
  - Positively skewed (right-tailed). Consider log-normal or gamma distribution.
- Jitter: Not normally distributed. Consider a custom distribution or transformation.
  - Negatively skewed (left-tailed). Consider beta or reflected distributions.
- Qoe: Not normally distributed. Consider a custom distribution or transformation.
  - Negatively skewed (left-tailed). Consider beta or reflected distributions.

### 2. Parameter Correlations

Maintain these significant correlations:

- Throughput and Jitter: negative correlation (-0.63)
- Packets_lost and Packet_loss_rate: positive correlation (0.94)
- Packets_lost and Qoe: negative correlation (-0.37)
- Packet_loss_rate and Qoe: negative correlation (-0.36)

### 3. QoE Calculation

Consider a more complex model for QoE calculation as the linear regression model is insufficient.
Possible approaches:

1. Use a non-linear model (polynomial regression, random forest, etc.)
2. Include interaction terms between parameters
3. Transform the parameters (log, square root, etc.) before regression

### 4. Temporal Patterns

Incorporate these temporal patterns:

- **Hour of Day**: Apply hourly pattern modifiers
- **Day of Week**: Apply day-of-week pattern modifiers

Recommended network condition profiles for different times:

1. **Workday Morning (7-9 AM)**: Higher network usage, potentially congested
2. **Workday Business Hours (9 AM-5 PM)**: Stable network conditions
3. **Workday Evening Peak (5-8 PM)**: Highest network usage, potentially congested
4. **Night (10 PM-6 AM)**: Low network usage, better conditions
5. **Weekend**: Different pattern from weekdays

### 5. Anomalies

The original data contains 941 anomalies out of 10925 measurements (8.61%).

For realistic synthetic data, occasionally introduce anomalies at a similar rate.

### 6. Filling Time Gaps

When filling the identified time gaps:

1. **Maintain proper timestamp sequencing** with correct 2-second intervals within files
2. **Group measurements into 10-second windows** matching the original file structure
3. **Preserve parameter correlations** as identified above
4. **Apply appropriate temporal patterns** based on time of day and day of week
5. **Calculate QoE** using the recommended model or formula
6. **Ensure smooth transitions** at the boundaries between original and synthetic data
