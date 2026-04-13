# ============================================================
# NASA CMAPSS Turbofan Engine — Predictive Maintenance
# Notebook Generator
#
# This creates a complete Jupyter notebook for the real
# NASA turbofan engine degradation dataset.
#
# Run:  python notebooks/create_nasa_notebook.py
# Then open: notebooks/nasa_cmapss_analysis.ipynb
# ============================================================

import json
import os

cells = [

# ── 0. Title ──────────────────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"# NASA CMAPSS Turbofan Engine — Predictive Maintenance\n",
"\n",
"**Dataset:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)  \n",
"**Task:** Predict **Remaining Useful Life (RUL)** of jet engines  \n",
"**Domain:** Aviation | Aerospace | Industrial IoT\n",
"\n",
"---\n",
"\n",
"## What is CMAPSS?\n",
"The CMAPSS dataset simulates jet turbofan engine degradation.  \n",
"Each engine starts healthy and degrades over time until failure.  \n",
"We train a model to predict **how many cycles remain** before breakdown.\n",
"\n",
"## Dataset Files (place in `data/nasa_cmapss/`)\n",
"| File | Contents |\n",
"|------|----------|\n",
"| `train_FD001.txt` | Training engine sensor readings |\n",
"| `test_FD001.txt` | Test engine sensor readings |\n",
"| `RUL_FD001.txt` | True RUL values for test engines |\n",
"\n",
"## Column Layout\n",
"| Columns | Meaning |\n",
"|---------|--------|\n",
"| 1 | Engine unit ID |\n",
"| 2 | Time cycle |\n",
"| 3–5 | Operational settings |\n",
"| 6–26 | Sensor measurements (s1–s21) |"
]
},

# ── 1. Imports ────────────────────────────────────────────────────
{
"cell_type":"markdown",
"source":["## Step 1: Install & Import Libraries"]
},
{
"cell_type":"code",
"source":[
"import numpy as np\n",
"import pandas as pd\n",
"import matplotlib.pyplot as plt\n",
"import seaborn as sns\n",
"import warnings\n",
"warnings.filterwarnings('ignore')\n",
"\n",
"from sklearn.preprocessing import MinMaxScaler\n",
"from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
"from sklearn.linear_model import LinearRegression\n",
"from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
"from sklearn.model_selection import train_test_split\n",
"import joblib, os\n",
"\n",
"print('All imports successful!')\n",
"print(f'NumPy: {np.__version__}, Pandas: {pd.__version__}')"
]
},

# ── 2. Load Data ──────────────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"## Step 2: Load NASA CMAPSS Dataset\n",
"\n",
"> Make sure `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`  \n",
"> are placed in the `data/nasa_cmapss/` folder."
]
},
{
"cell_type":"code",
"source":[
"# Column names (NASA CMAPSS has no header)\n",
"col_names = (\n",
"    ['unit_id', 'time_cycles']\n",
"    + [f'op_setting_{i}' for i in range(1, 4)]\n",
"    + [f's{i}' for i in range(1, 22)]\n",
")\n",
"\n",
"DATA_DIR = '../data/nasa_cmapss/'\n",
"\n",
"# Load training data\n",
"train_df = pd.read_csv(\n",
"    DATA_DIR + 'train_FD001.txt',\n",
"    sep=r'\\s+', header=None, names=col_names\n",
")\n",
"\n",
"# Load test data\n",
"test_df = pd.read_csv(\n",
"    DATA_DIR + 'test_FD001.txt',\n",
"    sep=r'\\s+', header=None, names=col_names\n",
")\n",
"\n",
"# Load true RUL values for test set\n",
"rul_df = pd.read_csv(\n",
"    DATA_DIR + 'RUL_FD001.txt',\n",
"    sep=r'\\s+', header=None, names=['true_rul']\n",
")\n",
"\n",
"print(f'Train shape: {train_df.shape}')\n",
"print(f'Test shape:  {test_df.shape}')\n",
"print(f'RUL shape:   {rul_df.shape}')\n",
"train_df.head()"
]
},

# ── 3. EDA ────────────────────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"## Step 3: Exploratory Data Analysis\n",
"\n",
"### Dataset Summary\n",
"- How many unique engines?\n",
"- How long do engines last on average?\n",
"- Are there any constant (useless) sensor columns?"
]
},
{
"cell_type":"code",
"source":[
"print('=== TRAINING DATA OVERVIEW ===')\n",
"print(f'Number of engines : {train_df[\"unit_id\"].nunique()}')\n",
"print(f'Total rows        : {len(train_df):,}')\n",
"print(f'\\nEngine lifetime stats (cycles):')\n",
"lifetime = train_df.groupby('unit_id')['time_cycles'].max()\n",
"print(lifetime.describe().round(1))\n",
"\n",
"# Histogram of engine lifetimes\n",
"plt.figure(figsize=(10, 4))\n",
"plt.hist(lifetime, bins=20, color='#00d4ff', edgecolor='white', alpha=0.8)\n",
"plt.xlabel('Total Cycles before Failure')\n",
"plt.ylabel('Number of Engines')\n",
"plt.title('Distribution of Engine Lifetimes (NASA FD001)', fontweight='bold')\n",
"plt.grid(alpha=0.3)\n",
"plt.tight_layout()\n",
"os.makedirs('../outputs/images', exist_ok=True)\n",
"plt.savefig('../outputs/images/nasa_01_engine_lifetimes.png', dpi=120, bbox_inches='tight')\n",
"plt.show()"
]
},
{
"cell_type":"code",
"source":[
"# Identify and drop sensors with near-zero variance (useless columns)\n",
"sensor_cols = [f's{i}' for i in range(1, 22)]\n",
"std_vals = train_df[sensor_cols].std()\n",
"\n",
"# Sensors with std < 0.01 carry no information\n",
"drop_sensors = std_vals[std_vals < 0.01].index.tolist()\n",
"drop_settings = ['op_setting_3']  # also often constant\n",
"drop_cols = drop_sensors + drop_settings\n",
"\n",
"print(f'Dropping {len(drop_cols)} constant/useless columns:')\n",
"print(drop_cols)\n",
"\n",
"train_df = train_df.drop(columns=drop_cols)\n",
"test_df  = test_df.drop(columns=drop_cols)\n",
"\n",
"# Update sensor column list\n",
"sensor_cols = [c for c in train_df.columns \n",
"               if c.startswith('s') and c not in drop_cols]\n",
"print(f'\\nRemaining useful sensor columns: {len(sensor_cols)}')\n",
"print(sensor_cols)"
]
},
{
"cell_type":"code",
"source":[
"# Plot sensor readings for one engine\n",
"engine_1 = train_df[train_df['unit_id'] == 1]\n",
"\n",
"# Show first 6 useful sensors\n",
"fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)\n",
"colors = ['#00d4ff','#ff4757','#2ed573','#ffa502','#a29bfe','#fd79a8']\n",
"\n",
"for ax, col, color in zip(axes.flat, sensor_cols[:6], colors):\n",
"    ax.plot(engine_1['time_cycles'], engine_1[col], color=color, linewidth=1)\n",
"    ax.set_ylabel(col, fontsize=9)\n",
"    ax.set_title(f'Sensor: {col}', fontsize=10)\n",
"    ax.grid(alpha=0.3)\n",
"\n",
"plt.suptitle('Engine #1 — Sensor Readings Over Operational Cycles', fontweight='bold')\n",
"plt.xlabel('Time Cycles')\n",
"plt.tight_layout()\n",
"plt.savefig('../outputs/images/nasa_02_sensor_readings.png', dpi=120, bbox_inches='tight')\n",
"plt.show()"
]
},

# ── 4. RUL Label ──────────────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"## Step 4: Create RUL (Remaining Useful Life) Label\n",
"\n",
"> **RUL** = how many more cycles until the engine fails.\n",
">\n",
"> Formula: `RUL = max_cycles_for_engine - current_cycle`\n",
">\n",
"> We also use a **piece-wise linear RUL** (cap at 125 cycles):  \n",
"> Early in engine life, RUL is capped. This is industry-standard  \n",
"> because we don't care about far-future predictions exactly."
]
},
{
"cell_type":"code",
"source":[
"RUL_CAP = 125  # standard CMAPSS cap\n",
"\n",
"# For training: compute RUL from max cycle of each engine\n",
"max_cycles = train_df.groupby('unit_id')['time_cycles'].max().reset_index()\n",
"max_cycles.columns = ['unit_id', 'max_cycles']\n",
"\n",
"train_df = train_df.merge(max_cycles, on='unit_id')\n",
"train_df['rul_raw'] = train_df['max_cycles'] - train_df['time_cycles']\n",
"train_df['rul']     = train_df['rul_raw'].clip(upper=RUL_CAP)  # piecewise linear\n",
"train_df = train_df.drop(columns=['max_cycles', 'rul_raw'])\n",
"\n",
"print(f'RUL range in training data: {train_df[\"rul\"].min()} → {train_df[\"rul\"].max()}')\n",
"\n",
"# Distribution of RUL\n",
"plt.figure(figsize=(10, 4))\n",
"plt.hist(train_df['rul'], bins=50, color='#ff4757', edgecolor='white', alpha=0.8)\n",
"plt.xlabel('Remaining Useful Life (cycles)')\n",
"plt.ylabel('Count')\n",
"plt.title('Distribution of RUL Labels (capped at 125)', fontweight='bold')\n",
"plt.grid(alpha=0.3)\n",
"plt.tight_layout()\n",
"plt.savefig('../outputs/images/nasa_03_rul_distribution.png', dpi=120, bbox_inches='tight')\n",
"plt.show()"
]
},
{
"cell_type":"code",
"source":[
"# Also create binary failure label (RUL <= 30 cycles = imminent failure)\n",
"FAILURE_THRESHOLD = 30\n",
"train_df['failure_imminent'] = (train_df['rul'] <= FAILURE_THRESHOLD).astype(int)\n",
"\n",
"print(f'Imminent failure label distribution:')\n",
"print(train_df['failure_imminent'].value_counts())\n",
"print(f'Failure rate: {train_df[\"failure_imminent\"].mean()*100:.1f}%')"
]
},

# ── 5. Feature Engineering ────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"## Step 5: Feature Engineering\n",
"\n",
"We create rolling window features to capture degradation trends:  \n",
"- Rolling mean (smoothed trend)\n",
"- Rolling std (variability / instability)"
]
},
{
"cell_type":"code",
"source":[
"def add_rolling_features_nasa(df, sensor_cols, windows=[5, 10]):\n",
"    \"\"\"Add rolling stats per engine (not across engines).\"\"\"\n",
"    df = df.sort_values(['unit_id', 'time_cycles']).copy()\n",
"    for col in sensor_cols:\n",
"        for w in windows:\n",
"            grp = df.groupby('unit_id')[col]\n",
"            df[f'{col}_rmean_{w}'] = grp.transform(\n",
"                lambda x: x.rolling(w, min_periods=1).mean())\n",
"            df[f'{col}_rstd_{w}'] = grp.transform(\n",
"                lambda x: x.rolling(w, min_periods=1).std().fillna(0))\n",
"    return df\n",
"\n",
"train_df = add_rolling_features_nasa(train_df, sensor_cols)\n",
"test_df  = add_rolling_features_nasa(test_df,  sensor_cols)\n",
"\n",
"print(f'Training feature count: {train_df.shape[1]}')\n",
"print(f'Training rows:          {len(train_df):,}')"
]
},

# ── 6. Preprocessing ──────────────────────────────────────────────
{
"cell_type":"markdown",
"source":["## Step 6: Scale Features & Prepare Train/Test"]
},
{
"cell_type":"code",
"source":[
"# Feature columns = everything except IDs, cycles, and labels\n",
"exclude = ['unit_id','time_cycles','rul','failure_imminent']\n",
"feature_cols = [c for c in train_df.columns if c not in exclude]\n",
"\n",
"X_train = train_df[feature_cols].values\n",
"y_rul   = train_df['rul'].values\n",
"y_fail  = train_df['failure_imminent'].values\n",
"\n",
"# For test: take last reading of each engine (that's what we predict)\n",
"test_last = test_df.groupby('unit_id').last().reset_index()\n",
"X_test    = test_last[feature_cols].values\n",
"\n",
"# Scale\n",
"scaler   = MinMaxScaler()\n",
"X_train  = scaler.fit_transform(X_train)\n",
"X_test   = scaler.transform(X_test)\n",
"\n",
"# True RUL for test engines\n",
"y_test_rul = rul_df['true_rul'].values.clip(0, RUL_CAP)\n",
"\n",
"print(f'X_train: {X_train.shape}  |  y_rul: {y_rul.shape}')\n",
"print(f'X_test:  {X_test.shape}   |  y_test: {y_test_rul.shape}')"
]
},

# ── 7. Model Training ─────────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"## Step 7: Train RUL Prediction Models\n",
"\n",
"We train **regression** models to predict exact RUL value:  \n",
"1. Linear Regression (baseline)\n",
"2. Random Forest Regressor\n",
"3. Gradient Boosting Regressor"
]
},
{
"cell_type":"code",
"source":[
"from sklearn.linear_model import LinearRegression\n",
"from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
"\n",
"models = {\n",
"    'Linear Regression'   : LinearRegression(),\n",
"    'Random Forest'       : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),\n",
"    'Gradient Boosting'   : GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),\n",
"}\n",
"\n",
"results = {}\n",
"for name, model in models.items():\n",
"    print(f'Training {name}...')\n",
"    model.fit(X_train, y_rul)\n",
"    y_pred = model.predict(X_test)\n",
"    y_pred = np.clip(y_pred, 0, RUL_CAP)  # clip to valid range\n",
"    \n",
"    rmse = np.sqrt(mean_squared_error(y_test_rul, y_pred))\n",
"    mae  = mean_absolute_error(y_test_rul, y_pred)\n",
"    r2   = r2_score(y_test_rul, y_pred)\n",
"    \n",
"    results[name] = {'model': model, 'pred': y_pred, 'rmse': rmse, 'mae': mae, 'r2': r2}\n",
"    print(f'  RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R2: {r2:.3f}')\n",
"    print()\n",
"\n",
"# Best model = lowest RMSE\n",
"best_name = min(results, key=lambda k: results[k]['rmse'])\n",
"best = results[best_name]\n",
"print(f'Best Model: {best_name}  (RMSE={best[\"rmse\"]:.2f})')"
]
},

# ── 8. Evaluation ─────────────────────────────────────────────────
{
"cell_type":"markdown",
"source":["## Step 8: Evaluate & Visualize Results"]
},
{
"cell_type":"code",
"source":[
"# Model comparison bar chart\n",
"fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
"\n",
"names = list(results.keys())\n",
"rmses = [results[n]['rmse'] for n in names]\n",
"r2s   = [results[n]['r2']   for n in names]\n",
"\n",
"axes[0].bar(names, rmses, color=['#00d4ff','#2ed573','#ffa502'], edgecolor='white')\n",
"axes[0].set_title('RMSE Comparison (lower = better)', fontweight='bold')\n",
"axes[0].set_ylabel('RMSE (cycles)')\n",
"axes[0].grid(axis='y', alpha=0.3)\n",
"for i,v in enumerate(rmses):\n",
"    axes[0].text(i, v+0.5, f'{v:.1f}', ha='center', fontweight='bold')\n",
"\n",
"axes[1].bar(names, r2s, color=['#00d4ff','#2ed573','#ffa502'], edgecolor='white')\n",
"axes[1].set_title('R² Score (higher = better)', fontweight='bold')\n",
"axes[1].set_ylabel('R² Score')\n",
"axes[1].set_ylim(0, 1.1)\n",
"axes[1].axhline(1.0, color='gray', linestyle='--', linewidth=0.8)\n",
"axes[1].grid(axis='y', alpha=0.3)\n",
"\n",
"plt.suptitle('NASA CMAPSS — RUL Prediction Model Comparison', fontweight='bold')\n",
"plt.tight_layout()\n",
"plt.savefig('../outputs/images/nasa_04_model_comparison.png', dpi=120, bbox_inches='tight')\n",
"plt.show()"
]
},
{
"cell_type":"code",
"source":[
"# Predicted vs Actual RUL scatter\n",
"best_pred = best['pred']\n",
"\n",
"plt.figure(figsize=(8, 6))\n",
"plt.scatter(y_test_rul, best_pred, alpha=0.6, color='#00d4ff', edgecolors='white', linewidths=0.3)\n",
"plt.plot([0, RUL_CAP], [0, RUL_CAP], 'r--', linewidth=1.5, label='Perfect prediction')\n",
"plt.xlabel('True RUL (cycles)', fontsize=11)\n",
"plt.ylabel('Predicted RUL (cycles)', fontsize=11)\n",
"plt.title(f'{best_name} — Predicted vs Actual RUL', fontweight='bold')\n",
"plt.legend()\n",
"plt.grid(alpha=0.3)\n",
"plt.tight_layout()\n",
"plt.savefig('../outputs/images/nasa_05_predicted_vs_actual.png', dpi=120, bbox_inches='tight')\n",
"plt.show()\n",
"print(f'Best model RMSE: {best[\"rmse\"]:.2f} cycles')"
]
},
{
"cell_type":"code",
"source":[
"# RUL prediction over engine lifetime (one engine)\n",
"engine_id = 1\n",
"eng = train_df[train_df['unit_id'] == engine_id].copy()\n",
"eng_X = scaler.transform(eng[feature_cols].values)\n",
"eng_pred_rul = best['model'].predict(eng_X)\n",
"eng_pred_rul = np.clip(eng_pred_rul, 0, RUL_CAP)\n",
"\n",
"plt.figure(figsize=(12, 5))\n",
"plt.plot(eng['time_cycles'], eng['rul'],      color='#2ed573', linewidth=2, label='True RUL')\n",
"plt.plot(eng['time_cycles'], eng_pred_rul,    color='#ff4757', linewidth=1.5,\n",
"         linestyle='--', label='Predicted RUL')\n",
"plt.axhline(FAILURE_THRESHOLD, color='orange', linestyle=':', linewidth=1.5,\n",
"            label=f'Alert Zone (<={FAILURE_THRESHOLD} cycles)')\n",
"plt.fill_between(eng['time_cycles'], 0, FAILURE_THRESHOLD, alpha=0.15, color='red')\n",
"plt.xlabel('Operational Cycle')\n",
"plt.ylabel('Remaining Useful Life (cycles)')\n",
"plt.title(f'Engine #{engine_id} — RUL Over Time', fontweight='bold')\n",
"plt.legend()\n",
"plt.grid(alpha=0.3)\n",
"plt.tight_layout()\n",
"plt.savefig('../outputs/images/nasa_06_rul_over_time.png', dpi=120, bbox_inches='tight')\n",
"plt.show()"
]
},
{
"cell_type":"code",
"source":[
"# Feature importance\n",
"if hasattr(best['model'], 'feature_importances_'):\n",
"    importances = best['model'].feature_importances_\n",
"    indices = np.argsort(importances)[::-1][:20]\n",
"    top_feats = [feature_cols[i] for i in indices]\n",
"    top_vals  = importances[indices]\n",
"\n",
"    plt.figure(figsize=(10, 7))\n",
"    colors_bar = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_feats)))\n",
"    plt.barh(range(len(top_feats)), top_vals[::-1], color=colors_bar[::-1])\n",
"    plt.yticks(range(len(top_feats)), top_feats[::-1], fontsize=9)\n",
"    plt.xlabel('Feature Importance')\n",
"    plt.title(f'{best_name} — Top 20 Feature Importances', fontweight='bold')\n",
"    plt.grid(axis='x', alpha=0.3)\n",
"    plt.tight_layout()\n",
"    plt.savefig('../outputs/images/nasa_07_feature_importance.png', dpi=120, bbox_inches='tight')\n",
"    plt.show()"
]
},

# ── 9. Alert System ───────────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"## Step 9: Maintenance Alert System\n",
"\n",
"Classify each test engine into a **risk category** based on predicted RUL."
]
},
{
"cell_type":"code",
"source":[
"def classify_risk(rul_val):\n",
"    if rul_val <= 15:  return 'CRITICAL'\n",
"    if rul_val <= 30:  return 'HIGH'\n",
"    if rul_val <= 60:  return 'MEDIUM'\n",
"    return 'LOW'\n",
"\n",
"alert_df = pd.DataFrame({\n",
"    'engine_id'   : test_last['unit_id'].values,\n",
"    'pred_rul'    : best_pred.round(1),\n",
"    'true_rul'    : y_test_rul,\n",
"    'risk_level'  : [classify_risk(r) for r in best_pred],\n",
"})\n",
"alert_df = alert_df.sort_values('pred_rul')\n",
"\n",
"print('=== MAINTENANCE ALERT SUMMARY ===')\n",
"print(alert_df['risk_level'].value_counts())\n",
"print('\\nCRITICAL + HIGH risk engines:')\n",
"critical = alert_df[alert_df['risk_level'].isin(['CRITICAL','HIGH'])]\n",
"print(critical.to_string(index=False))\n",
"\n",
"alert_df.to_csv('../outputs/nasa_alerts.csv', index=False)\n",
"print('\\nAlerts saved -> outputs/nasa_alerts.csv')"
]
},
{
"cell_type":"code",
"source":[
"# Risk distribution pie chart\n",
"risk_counts = alert_df['risk_level'].value_counts()\n",
"colors_pie  = {'CRITICAL':'#ff4757','HIGH':'#ffa502','MEDIUM':'#ffd32a','LOW':'#2ed573'}\n",
"pie_colors  = [colors_pie.get(l,'gray') for l in risk_counts.index]\n",
"\n",
"plt.figure(figsize=(7, 6))\n",
"wedges, texts, autotexts = plt.pie(\n",
"    risk_counts, labels=risk_counts.index,\n",
"    colors=pie_colors, autopct='%1.1f%%',\n",
"    startangle=140, pctdistance=0.8,\n",
"    wedgeprops={'edgecolor':'white','linewidth':1.5}\n",
")\n",
"for at in autotexts: at.set_fontsize(11)\n",
"plt.title('Engine Fleet — Risk Level Distribution', fontweight='bold', fontsize=13)\n",
"plt.tight_layout()\n",
"plt.savefig('../outputs/images/nasa_08_risk_distribution.png', dpi=120, bbox_inches='tight')\n",
"plt.show()"
]
},

# ── 10. Save model ────────────────────────────────────────────────
{
"cell_type":"code",
"source":[
"os.makedirs('../models', exist_ok=True)\n",
"joblib.dump(best['model'], '../models/nasa_rul_model.joblib')\n",
"joblib.dump(scaler,        '../models/nasa_scaler.joblib')\n",
"joblib.dump(feature_cols,  '../models/nasa_feature_cols.joblib')\n",
"print('Model saved to models/nasa_rul_model.joblib')\n",
"print('Scaler saved to models/nasa_scaler.joblib')"
]
},

# ── 11. Conclusion ────────────────────────────────────────────────
{
"cell_type":"markdown",
"source":[
"## Conclusion\n",
"\n",
"### What We Built with Real NASA Data\n",
"- Loaded and cleaned the NASA CMAPSS FD001 dataset\n",
"- Dropped constant/useless sensor columns automatically\n",
"- Created RUL (Remaining Useful Life) regression labels\n",
"- Built rolling window features for trend detection\n",
"- Trained 3 regression models and compared them\n",
"- Built a fleet-level risk classification alert system\n",
"- Generated 8 production-quality charts\n",
"\n",
"### Key Metrics (FD001 Typical Results)\n",
"| Model | Typical RMSE |\n",
"|-------|--------------|\n",
"| Linear Regression | 28–35 cycles |\n",
"| Random Forest | 18–24 cycles |\n",
"| Gradient Boosting | 16–22 cycles |\n",
"\n",
"### Next Steps\n",
"- Try LSTM for sequence-based RUL prediction\n",
"- Use all 4 FD sub-datasets (FD001–FD004)\n",
"- Deploy as a FastAPI prediction endpoint"
]
},

] # end cells

# ── Build notebook JSON ───────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.11.0"},
    },
    "cells": [],
}

for i, cell in enumerate(cells):
    if cell["cell_type"] == "markdown":
        notebook["cells"].append({
            "cell_type": "markdown",
            "id": f"cell-{i}",
            "metadata": {},
            "source": cell["source"],
        })
    else:
        notebook["cells"].append({
            "cell_type": "code",
            "id": f"cell-{i}",
            "metadata": {},
            "source": cell["source"],
            "outputs": [],
            "execution_count": None,
        })

out_path = os.path.join(os.path.dirname(__file__), "nasa_cmapss_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"[OK] NASA notebook created: {out_path}")
