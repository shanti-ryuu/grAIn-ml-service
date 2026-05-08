"""
Synthetic Rice Grain Drying Data Generator
==========================================
Generates training data based on Page's thin-layer drying equation:
    MR = exp(-k * t^n)

Where:
    MR = Moisture Ratio = (M - Me) / (M0 - Me)
    M  = Current moisture content (% wet basis)
    M0 = Initial moisture content
    Me = Equilibrium moisture content
    k  = Drying constant (depends on temperature, airflow)
    n  = Page's exponent (typically 0.8-1.2 for rice)
    t  = Drying time (minutes)

References:
- Midilli et al. (2002) - Drying kinetics of rice
- Brooker et al. (1992) - Drying and Storage of Grains
- Henderson & Pabis model variations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# Physical constants for rice drying
RICE_PARAMS = {
    "initial_moisture_range": (18.0, 28.0),   # % wet basis (freshly harvested)
    "target_moisture": 14.0,                    # % wet basis (storage safe)
    "equilibrium_moisture_range": (8.0, 12.0),  # % wet basis (depends on RH)
    "k_base": 0.0025,                           # Base drying constant
    "n_range": (0.85, 1.15),                    # Page's exponent range
    "temp_range": (35.0, 70.0),                 # Drying air temperature (°C)
    "humidity_range": (30.0, 85.0),             # Ambient relative humidity (%)
    "fan_speed_range": (30.0, 100.0),           # Fan speed (%)
    "solar_voltage_range": (0.0, 24.0),         # Solar panel voltage (V)
    "weight_range": (5.0, 50.0),                # Grain weight (kg)
}


def calculate_equilibrium_moisture(temperature: float, humidity: float) -> float:
    """
    Henderson-Thompson equation for equilibrium moisture content of rice.
    Me = [ln(1 - RH) / (-A * (T + C))]^(1/B)

    Simplified version with empirical coefficients for rough rice.
    """
    rh = humidity / 100.0
    rh = np.clip(rh, 0.05, 0.95)

    A = 5.684e-5
    B = 1.9187
    C = 49.810

    me = (np.log(1.0 - rh) / (-A * (temperature + C))) ** (1.0 / B)
    return np.clip(me, 6.0, 16.0)


def calculate_drying_constant(temperature: float, fan_speed: float,
                               solar_voltage: float, humidity: float) -> float:
    """
    Calculate drying constant k based on environmental conditions.
    Higher temperature and fan speed increase drying rate.
    Higher humidity decreases drying rate.
    Solar energy provides supplemental heating.
    """
    temp_factor = np.exp(0.035 * (temperature - 40.0))
    fan_factor = 0.4 + 0.6 * (fan_speed / 100.0)
    humidity_factor = 1.0 - 0.4 * ((humidity - 30.0) / 70.0)
    solar_factor = 1.0 + 0.15 * (solar_voltage / 24.0)

    k = RICE_PARAMS["k_base"] * temp_factor * fan_factor * humidity_factor * solar_factor
    noise = np.random.normal(1.0, 0.05)
    return k * noise


def generate_single_session(session_id: int) -> pd.DataFrame:
    """Generate a complete drying session with realistic sensor readings."""

    # Initial conditions (randomized)
    initial_moisture = np.random.uniform(*RICE_PARAMS["initial_moisture_range"])
    temperature_base = np.random.uniform(*RICE_PARAMS["temp_range"])
    humidity_base = np.random.uniform(*RICE_PARAMS["humidity_range"])
    fan_speed_base = np.random.uniform(*RICE_PARAMS["fan_speed_range"])
    solar_voltage_base = np.random.uniform(8.0, 22.0)
    grain_weight = np.random.uniform(*RICE_PARAMS["weight_range"])
    n = np.random.uniform(*RICE_PARAMS["n_range"])

    # Determine equilibrium moisture
    me = calculate_equilibrium_moisture(temperature_base, humidity_base)

    # Generate time series (readings every 30 seconds, session can last 2-8 hours)
    max_duration_minutes = np.random.uniform(120, 480)
    reading_interval = 0.5  # minutes (30 seconds)

    records = []
    t = 0.0
    current_moisture = initial_moisture
    energy_consumed = 0.0

    while t < max_duration_minutes and current_moisture > RICE_PARAMS["target_moisture"] - 1.0:
        # Add environmental variation over time
        time_hours = t / 60.0

        # Temperature fluctuates ±3°C with slight trend
        temp = temperature_base + np.random.normal(0, 1.5) + 2.0 * np.sin(time_hours * 0.5)
        temp = np.clip(temp, 30.0, 75.0)

        # Humidity varies ±5% with inverse temperature correlation
        humid = humidity_base + np.random.normal(0, 2.5) - 0.3 * (temp - temperature_base)
        humid = np.clip(humid, 20.0, 95.0)

        # Fan speed with occasional adjustments
        fan = fan_speed_base + np.random.normal(0, 3.0)
        fan = np.clip(fan, 20.0, 100.0)

        # Solar voltage follows daily pattern (lower in morning/evening)
        solar = solar_voltage_base * (0.7 + 0.3 * np.sin(np.pi * time_hours / 8.0))
        solar += np.random.normal(0, 1.0)
        solar = np.clip(solar, 0.0, 24.0)

        # Calculate current drying constant
        k = calculate_drying_constant(temp, fan, solar, humid)
        me_current = calculate_equilibrium_moisture(temp, humid)

        # Page's equation: MR = exp(-k * t^n)
        if t > 0:
            mr = np.exp(-k * (t ** n))
            current_moisture = me_current + (initial_moisture - me_current) * mr
            # Add measurement noise
            current_moisture += np.random.normal(0, 0.15)
            current_moisture = max(current_moisture, me_current)

        # Energy consumption (kWh) - fans + heaters
        power_kw = (0.2 + 0.8 * (fan / 100.0)) * (1.0 + 0.5 * (temp - 35.0) / 35.0)
        energy_consumed += power_kw * (reading_interval / 60.0)

        # Weight decreases as moisture is removed
        moisture_lost_fraction = (initial_moisture - current_moisture) / 100.0
        current_weight = grain_weight * (1.0 - moisture_lost_fraction * 0.5)

        # Determine dryer status
        if current_moisture <= RICE_PARAMS["target_moisture"]:
            status = "completed"
        elif temp > 68.0:
            status = "warning"
        else:
            status = "running"

        records.append({
            "session_id": session_id,
            "timestamp_minutes": round(t, 1),
            "temperature": round(temp, 2),
            "humidity": round(humid, 2),
            "moisture": round(current_moisture, 2),
            "fan_speed": round(fan, 1),
            "solar_voltage": round(solar, 2),
            "energy_consumed": round(energy_consumed, 4),
            "weight": round(current_weight, 2),
            "initial_moisture": round(initial_moisture, 2),
            "target_moisture": RICE_PARAMS["target_moisture"],
            "grain_type": "rice",
            "status": status,
        })

        t += reading_interval

        # Occasionally simulate sensor gaps (missing readings)
        if np.random.random() < 0.02:
            t += reading_interval * np.random.randint(1, 4)

    return pd.DataFrame(records)


def generate_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw session data into ML training features.
    For each reading, predict what moisture will be in 30 minutes.
    """
    training_rows = []

    for session_id in df["session_id"].unique():
        session = df[df["session_id"] == session_id].reset_index(drop=True)

        for i in range(len(session)):
            current = session.iloc[i]
            current_time = current["timestamp_minutes"]

            # Find the reading closest to 30 minutes in the future
            future_mask = session["timestamp_minutes"] >= current_time + 28.0
            future_mask &= session["timestamp_minutes"] <= current_time + 32.0
            future_readings = session[future_mask]

            if len(future_readings) == 0:
                # Try wider window
                future_mask2 = session["timestamp_minutes"] >= current_time + 25.0
                future_mask2 &= session["timestamp_minutes"] <= current_time + 35.0
                future_readings = session[future_mask2]

            if len(future_readings) == 0:
                continue

            future = future_readings.iloc[len(future_readings) // 2]

            # Calculate time to target
            target_readings = session[session["moisture"] <= RICE_PARAMS["target_moisture"]]
            if len(target_readings) > 0:
                time_to_target = target_readings.iloc[0]["timestamp_minutes"] - current_time
            else:
                time_to_target = (session.iloc[-1]["timestamp_minutes"] - current_time) * 1.5

            # Calculate recent drying rate (moisture change per minute over last 5 readings)
            if i >= 5:
                recent = session.iloc[i-5:i+1]
                drying_rate = (recent.iloc[0]["moisture"] - recent.iloc[-1]["moisture"]) / \
                             (recent.iloc[-1]["timestamp_minutes"] - recent.iloc[0]["timestamp_minutes"] + 0.01)
            else:
                drying_rate = 0.0

            # Moisture difference from target
            moisture_diff = current["moisture"] - RICE_PARAMS["target_moisture"]

            training_rows.append({
                # Input features
                "temperature": current["temperature"],
                "humidity": current["humidity"],
                "moisture": current["moisture"],
                "fan_speed": current["fan_speed"],
                "solar_voltage": current["solar_voltage"],
                "time_elapsed": current["timestamp_minutes"],
                "energy_consumed": current["energy_consumed"],
                "weight": current["weight"],
                "initial_moisture": current["initial_moisture"],
                "drying_rate": round(drying_rate, 6),
                "moisture_diff_from_target": round(moisture_diff, 2),

                # Target variables (what we predict)
                "moisture_30min": round(future["moisture"], 2),
                "time_to_target": round(max(0, time_to_target), 1),

                # Metadata
                "session_id": session_id,
                "is_drying_complete": current["moisture"] <= RICE_PARAMS["target_moisture"],
            })

    return pd.DataFrame(training_rows)


def main():
    print("=" * 60)
    print("grAIn Synthetic Data Generator")
    print("Rice Grain Drying - Page's Equation Model")
    print("=" * 60)

    num_sessions = 30  # Generate 30 complete drying sessions

    print(f"\nGenerating {num_sessions} synthetic drying sessions...")

    all_sessions = []
    for i in range(num_sessions):
        session_df = generate_single_session(i)
        all_sessions.append(session_df)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_sessions} sessions")

    raw_data = pd.concat(all_sessions, ignore_index=True)
    print(f"\nTotal raw sensor readings: {len(raw_data):,}")
    print(f"  Sessions: {raw_data['session_id'].nunique()}")
    print(f"  Avg readings per session: {len(raw_data) // num_sessions}")

    # Generate ML training features
    print("\nGenerating ML training features...")
    training_data = generate_training_features(raw_data)
    print(f"Total training samples: {len(training_data):,}")

    # Save datasets
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    raw_path = os.path.join(data_dir, "raw_sensor_data.csv")
    train_path = os.path.join(data_dir, "training_data.csv")

    raw_data.to_csv(raw_path, index=False)
    training_data.to_csv(train_path, index=False)

    print(f"\nSaved:")
    print(f"  Raw data: {raw_path} ({len(raw_data):,} rows)")
    print(f"  Training data: {train_path} ({len(training_data):,} rows)")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"\nFeature ranges:")
    print(f"  Temperature:    {training_data['temperature'].min():.1f} - {training_data['temperature'].max():.1f} °C")
    print(f"  Humidity:       {training_data['humidity'].min():.1f} - {training_data['humidity'].max():.1f} %")
    print(f"  Moisture:       {training_data['moisture'].min():.1f} - {training_data['moisture'].max():.1f} %")
    print(f"  Fan Speed:      {training_data['fan_speed'].min():.1f} - {training_data['fan_speed'].max():.1f} %")
    print(f"  Solar Voltage:  {training_data['solar_voltage'].min():.1f} - {training_data['solar_voltage'].max():.1f} V")
    print(f"  Time Elapsed:   {training_data['time_elapsed'].min():.1f} - {training_data['time_elapsed'].max():.1f} min")

    print(f"\nTarget ranges:")
    print(f"  Moisture 30min: {training_data['moisture_30min'].min():.1f} - {training_data['moisture_30min'].max():.1f} %")
    print(f"  Time to target: {training_data['time_to_target'].min():.1f} - {training_data['time_to_target'].max():.1f} min")

    print(f"\nDrying complete samples: {training_data['is_drying_complete'].sum()} / {len(training_data)}")
    print(f"  ({100 * training_data['is_drying_complete'].mean():.1f}% of dataset)")


if __name__ == "__main__":
    main()
