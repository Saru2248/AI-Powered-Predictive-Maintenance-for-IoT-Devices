# Architecture Deep-Dive
# AI-Powered Predictive Maintenance for IoT Devices

## System Architecture

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT LAYER (Sensors)                      │
│                                                              │
│   [Temperature]  [Vibration]  [Pressure]  [RPM]  [OilLevel]  │
│       °C           mm/s          PSI       RPM        %       │
│                                                              │
│           5 Machines × 60 days × 1 sample/hour               │
│                     = 7,200 rows/machine                      │
│                     = 36,000 total rows                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                PREPROCESSING MODULE                           │
│                                                              │
│  Step 1: Load CSV (parse timestamps)                         │
│  Step 2: Forward-fill missing values per machine             │
│  Step 3: Z-score outlier clipping (z > 3.5)                  │
│  Step 4: Physical range validation                           │
│           - RPM ≥ 0                                          │
│           - Oil 0–100%                                       │
│           - Vibration ≥ 0                                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│             FEATURE ENGINEERING MODULE                        │
│                                                              │
│  For each of 5 sensor columns:                               │
│                                                              │
│  Rolling stats (3h, 6h, 12h):                                │
│    → col_roll_mean_3h, col_roll_std_3h, col_roll_max_3h      │
│    → 5 sensors × 3 windows × 3 stats = 45 features           │
│                                                              │
│  Lag features (1h, 3h, 6h ago):                              │
│    → col_lag_1h, col_lag_3h, col_lag_6h                      │
│    → 5 sensors × 3 lags = 15 features                        │
│                                                              │
│  Rate of change:                                             │
│    → col_rate_of_change                                      │
│    → 5 features                                              │
│                                                              │
│  Interaction features (domain-driven):                       │
│    → temp_vib_stress, pressure_rpm_load, oil_ratio           │
│    → 3 features                                              │
│                                                              │
│  Time features:                                              │
│    → hour, day_of_week, month, is_weekend                    │
│    → 4 features                                              │
│                                                              │
│  TOTAL: 5 (raw) + 67 (engineered) ≈ 72 features              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING MODULE                        │
│                                                              │
│  Train/Test Split: 80% / 20% (stratified)                    │
│  Scaling: StandardScaler (mean=0, std=1)                     │
│  Balancing: SMOTE on training set                            │
│                                                              │
│  Models Trained:                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 1. Logistic Regression (baseline)                       │ │
│  │    max_iter=500, class_weight='balanced'                 │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │ 2. Random Forest (100 trees)                            │ │
│  │    n_estimators=100, max_depth=10                       │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │ 3. Gradient Boosting (100 estimators)                   │ │
│  │    learning_rate=0.1, max_depth=5                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  Selection criteria: Best F1 Score                           │
│  Saved to: models/best_model.joblib                          │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│             PREDICTION & ALERT MODULE                         │
│                                                              │
│  For each sensor reading:                                    │
│    1. Transform with saved scaler                            │
│    2. Predict failure probability (0.0 → 1.0)               │
│    3. Classify risk level:                                   │
│       < 0.25 → LOW                                           │
│       0.25–0.50 → MEDIUM                                     │
│       0.50–0.75 → HIGH                                       │
│       > 0.75 → CRITICAL                                      │
│                                                              │
│  Alert triggers (hybrid approach):                           │
│    ML Alert:    predicted_fail == 1                          │
│    Rule Alert:  temperature > 110°C                          │
│                 vibration > 2.0 mm/s                        │
│                 oil_level < 15%                              │
│                 rpm < 1000                                   │
│                                                              │
│  Output: alerts.csv + console report                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                 VISUALIZATION MODULE                          │
│                                                              │
│  01 — Sensor time-series (per machine)                       │
│  02 — Failure distribution by class & machine                │
│  03 — Correlation heatmap                                    │
│  04 — Model comparison bar chart                             │
│  05 — Confusion matrix                                       │
│  06 — ROC curve                                              │
│  07 — Feature importance (top 20)                            │
│  08 — Failure probability over time                          │
│  09 — Sensor box plots (normal vs failure)                   │
│  10 — Summary dashboard                                      │
│                                                              │
│  All saved to: outputs/images/                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|---------------|
| Data Generator | `src/data_generator.py` | Simulate IoT sensor readings with degradation |
| Preprocessor | `src/preprocessor.py` | Clean, validate, handle missing data |
| Feature Engineer | `src/feature_engineer.py` | Create 70+ predictive features |
| Model Trainer | `src/model_trainer.py` | Train, compare, evaluate, save best model |
| Predictor | `src/predictor.py` | Predict failures, classify risk, raise alerts |
| Visualizer | `src/visualizer.py` | Generate all charts and plots |
| Entry Point | `main.py` | Orchestrate the full pipeline |

---

## Key Design Decisions

### 1. Why SMOTE?
Failures are rare events (5–10% of readings). Without balancing:
- Model predicts "No Failure" always → 95% accuracy but useless!
- SMOTE creates synthetic minority (failure) samples so the model learns both classes equally.

### 2. Why F1 Score for model selection?
- In maintenance, both false alarms (precision) and missed failures (recall) matter.
- F1 = harmonic mean of precision and recall.
- For safety-critical systems, recall is weighted more (missing a failure = disastrous).

### 3. Why rolling features?
- A single temperature spike could be noise.
- A sustained temperature rise over 6 hours is a real signal.
- Rolling mean/std captures these sustained patterns.

### 4. Why lag features?
- ML models see each row independently.
- Lag features inject "memory" — the model sees past states.
- `temperature_lag_3h` = "what was the temperature 3 hours ago?"

### 5. Why hybrid alerts?
- ML alone may miss novel failure modes (unseen during training).
- Rule-based catches obvious physical violations (oil_level = 5%).
- Combined = more robust than either alone.
