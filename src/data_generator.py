# ============================================================
# Module: data_generator.py
# Purpose: Simulate realistic IoT sensor data for predictive
#          maintenance — mimics sensors on industrial machines
#          (motors, pumps, compressors, turbines).
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────
# SEED for reproducibility
# ─────────────────────────────────────────
np.random.seed(42)


def generate_sensor_data(
    n_machines: int = 5,
    days: int = 60,
    samples_per_hour: int = 1,
    failure_probability: float = 0.05,
    save_path: str = "data/sensor_data.csv",
) -> pd.DataFrame:
    """
    Generate synthetic IoT sensor readings for multiple machines.

    Each row represents one sensor reading for one machine at one timestamp.

    Parameters
    ----------
    n_machines        : int   – Number of machines to simulate
    days              : int   – Number of days of data to generate
    samples_per_hour  : int   – Readings per hour
    failure_probability: float – Base probability of failure label
    save_path         : str   – Where to save the CSV

    Returns
    
    -------
    pd.DataFrame with columns:
        machine_id, timestamp, temperature, vibration,
        pressure, rpm, oil_level, failure
    """

    total_samples = days * 24 * samples_per_hour
    records = []

    # ── Simulate each machine independently ──────────────────
    for machine_id in range(1, n_machines + 1):

        # Base "healthy" sensor values differ per machine
        base_temp       = np.random.uniform(60, 80)   # °C
        base_vibration  = np.random.uniform(0.2, 0.5) # mm/s
        base_pressure   = np.random.uniform(90, 110)  # PSI
        base_rpm        = np.random.uniform(1400, 1600)
        base_oil        = np.random.uniform(75, 95)   # % level

        # Start timestamp
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        for i in range(total_samples):

            timestamp = start_time + timedelta(hours=i / samples_per_hour)

            # ── Gradual degradation pattern ──────────────────
            # Every machine slowly degrades over time.
            # Degradation accelerates in the last 10% of life.
            degradation = i / total_samples  # 0.0 → 1.0
            accel        = max(0.0, (degradation - 0.7) * 3)  # kicks in late

            # ── Sensor readings with noise ────────────────────
            temperature = (
                base_temp
                + degradation * 25               # gradual heat rise
                + accel * 15                     # acceleration near failure
                + np.random.normal(0, 2)         # measurement noise
            )

            vibration = (
                base_vibration
                + degradation * 1.2
                + accel * 0.8
                + np.random.normal(0, 0.05)
            )
            vibration = max(0.0, vibration)

            pressure = (
                base_pressure
                + np.random.normal(0, 5)         # pressure fluctuates
                - degradation * 10               # slow drop under wear
                - accel * 8
            )

            rpm = (
                base_rpm
                - degradation * 100
                - accel * 50
                + np.random.normal(0, 20)
            )
            rpm = max(0.0, rpm)

            oil_level = (
                base_oil
                - degradation * 15               # oil depletes over time
                - accel * 8
                + np.random.normal(0, 1)
            )
            oil_level = max(0.0, min(100.0, oil_level))

            # ── Failure label ─────────────────────────────────
            # Failure more likely when sensors show extreme values
            dynamic_failure_prob = failure_probability + accel * 0.4
            failure = int(np.random.random() < dynamic_failure_prob)

            records.append({
                "machine_id" : machine_id,
                "timestamp"  : timestamp,
                "temperature": round(temperature, 2),
                "vibration"  : round(vibration, 4),
                "pressure"   : round(pressure, 2),
                "rpm"        : round(rpm, 1),
                "oil_level"  : round(oil_level, 2),
                "failure"    : failure,
            })

    df = pd.DataFrame(records)

    # ── Save to CSV ───────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[✓] Dataset generated:  {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"[✓] Saved to:           {save_path}")
    print(f"[✓] Failure rate:       {df['failure'].mean()*100:.2f}%")

    return df


# ─────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────
if __name__ == "__main__":
    df = generate_sensor_data(n_machines=5, days=60)
    print(df.head(10))
    print(df.describe())
