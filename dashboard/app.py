# ============================================================
# dashboard/app.py
# Flask backend — serves data from outputs/ as JSON APIs
# Run: python dashboard/app.py
# Open: http://localhost:5000
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(BASE, 'data',    'sensor_data.csv')
PRED_PATH = os.path.join(BASE, 'outputs', 'predictions.csv')
ALRT_PATH = os.path.join(BASE, 'outputs', 'alerts.csv')
IMG_DIR   = os.path.join(BASE, 'outputs', 'images')

# ── Cache loaded DataFrames ─────────────────────────────────
_cache = {}

def load_df(path, key):
    if key not in _cache:
        _cache[key] = pd.read_csv(path, parse_dates=['timestamp']
                                   if 'timestamp' in pd.read_csv(path, nrows=1).columns
                                   else None)
    return _cache[key]


# ========================
#   PAGE ROUTES
# ========================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)


# ========================
#   API ENDPOINTS
# ========================

@app.route('/api/summary')
def summary():
    """Top-level KPI cards."""
    try:
        pred = load_df(PRED_PATH, 'pred')
        alrt = load_df(ALRT_PATH, 'alrt') if os.path.exists(ALRT_PATH) else pd.DataFrame()
        raw  = load_df(DATA_PATH, 'raw')

        n_machines    = int(pred['machine_id'].nunique())
        total_alerts  = len(alrt)
        critical      = int((alrt['risk_level'] == 'CRITICAL').sum()) if not alrt.empty else 0
        high          = int((alrt['risk_level'] == 'HIGH').sum())    if not alrt.empty else 0
        failure_rate  = round(float(raw['failure'].mean()) * 100, 2)
        avg_fail_prob = round(float(pred['failure_prob'].mean()) * 100, 2)

        # Per-machine latest risk
        latest = (pred.sort_values('timestamp')
                      .groupby('machine_id')
                      .last()
                      .reset_index()[['machine_id','failure_prob','risk_level']])

        machine_status = []
        for _, row in latest.iterrows():
            machine_status.append({
                'id'       : int(row['machine_id']),
                'prob'     : round(float(row['failure_prob']) * 100, 1),
                'risk'     : row['risk_level'],
            })

        return jsonify({
            'n_machines'   : n_machines,
            'total_alerts' : total_alerts,
            'critical'     : critical,
            'high'         : high,
            'failure_rate' : failure_rate,
            'avg_fail_prob': avg_fail_prob,
            'machine_status': machine_status,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sensor_timeseries/<int:machine_id>')
def sensor_timeseries(machine_id):
    """Sensor readings over time for a given machine (sampled for speed)."""
    try:
        df  = load_df(DATA_PATH, 'raw')
        mdf = df[df['machine_id'] == machine_id].copy()

        # Sample every 6 hours to keep payload small
        mdf = mdf.iloc[::6].copy()

        return jsonify({
            'timestamps' : mdf['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'temperature': mdf['temperature'].round(2).tolist(),
            'vibration'  : mdf['vibration'].round(4).tolist(),
            'pressure'   : mdf['pressure'].round(2).tolist(),
            'rpm'        : mdf['rpm'].round(1).tolist(),
            'oil_level'  : mdf['oil_level'].round(2).tolist(),
            'failure'    : mdf['failure'].tolist(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/failure_prob/<int:machine_id>')
def failure_prob(machine_id):
    """Failure probability over time for a given machine (sampled)."""
    try:
        pred = load_df(PRED_PATH, 'pred')
        mdf  = pred[pred['machine_id'] == machine_id].iloc[::6].copy()

        return jsonify({
            'timestamps'  : mdf['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'failure_prob': (mdf['failure_prob'] * 100).round(1).tolist(),
            'risk_level'  : mdf['risk_level'].tolist(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts')
def alerts():
    """Top 50 alerts sorted by failure probability."""
    try:
        if not os.path.exists(ALRT_PATH):
            return jsonify([])
        alrt = load_df(ALRT_PATH, 'alrt')
        top  = alrt.head(50).copy()
        # Format timestamp
        if 'timestamp' in top.columns:
            top['timestamp'] = pd.to_datetime(top['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        top['failure_prob'] = (top['failure_prob'] * 100).round(1)
        return jsonify(top.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk_distribution')
def risk_distribution():
    """Count of alerts by risk level."""
    try:
        if not os.path.exists(ALRT_PATH):
            return jsonify({'LOW':0,'MEDIUM':0,'HIGH':0,'CRITICAL':0})
        alrt   = load_df(ALRT_PATH, 'alrt')
        counts = alrt['risk_level'].value_counts().to_dict()
        for lvl in ['LOW','MEDIUM','HIGH','CRITICAL']:
            counts.setdefault(lvl, 0)
        return jsonify(counts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sensor_stats')
def sensor_stats():
    """Box-plot style stats per sensor split by failure label."""
    try:
        df = load_df(DATA_PATH, 'raw')
        sensors = ['temperature','vibration','pressure','rpm','oil_level']
        result  = {}
        for s in sensors:
            normal  = df[df['failure']==0][s]
            failure = df[df['failure']==1][s]
            result[s] = {
                'normal' : {
                    'min': round(float(normal.min()),2),
                    'q1' : round(float(normal.quantile(.25)),2),
                    'med': round(float(normal.median()),2),
                    'q3' : round(float(normal.quantile(.75)),2),
                    'max': round(float(normal.max()),2),
                    'mean':round(float(normal.mean()),2),
                },
                'failure': {
                    'min': round(float(failure.min()),2),
                    'q1' : round(float(failure.quantile(.25)),2),
                    'med': round(float(failure.median()),2),
                    'q3' : round(float(failure.quantile(.75)),2),
                    'max': round(float(failure.max()),2),
                    'mean':round(float(failure.mean()),2),
                },
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/images')
def list_images():
    """Return list of available chart images."""
    try:
        files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
        return jsonify(files)
    except Exception as e:
        return jsonify([])


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  AI Predictive Maintenance Dashboard")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, host='0.0.0.0')
