import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_coordinates(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Normalize coordinates to [-1, 1] range using center-based scaling."""
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    scale = max(lat_max - lat_min, lon_max - lon_min) / 2
    if scale == 0:
        scale = 1.0

    lat_norm = (lat - lat_center) / scale
    lon_norm = (lon - lon_center) / scale
    return lat_norm, lon_norm, scale


def _create_spatial_features(lat_norm, lon_norm):
    """Create spatial feature matrix with sin/cos transformations."""
    return np.column_stack(
        [
            lat_norm,
            lon_norm,
            np.sin(2 * np.pi * lat_norm),
            np.cos(2 * np.pi * lat_norm),
            np.sin(2 * np.pi * lon_norm),
            np.cos(2 * np.pi * lon_norm),
        ]
    ).astype(np.float32)


def _save_dataframe(df, out_path):
    """Save DataFrame to CSV, creating directories if needed."""
    if out_path:
        out_dir = Path(out_path).parent
        if str(out_dir) != ".":
            out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved dataset to {out_path}")


def _generate_ground_truth_field(lat_norm, lon_norm, rng, spatial_noise_std=5.0):
    """Generate a weakly structured spatial field with additive non-spatial noise."""
    base = (
        0.3 * np.sin(2 * np.pi * lat_norm)
        + 0.25 * np.cos(2 * np.pi * lon_norm)
        + 0.1 * np.sin(2 * np.pi * lat_norm) * np.cos(2 * np.pi * lon_norm)
    )

    field = 15.0 + 5.0 * base

    for _ in range(2):
        cx, cy = rng.uniform(-0.3, 1.3, 2)
        dist2 = (lat_norm - cx) ** 2 + (lon_norm - cy) ** 2
        intensity = rng.uniform(2.0, 5.0)
        scale = rng.uniform(0.06, 0.15)
        field += intensity * np.exp(-dist2 / scale)

    field += rng.normal(0.0, spatial_noise_std, len(lat_norm))

    return np.maximum(0.0, field).astype(np.float32)


def _generate_sensors(y_true, lat_norm, lon_norm, rng, ambiguity_level):
    """Generate sensor readings with varying bias and noise profiles."""
    n_samples = len(y_true)
    base_ambiguity = float(ambiguity_level)

    # Sensor 0: most reliable
    scale_0 = 1.0 + rng.normal(0.0, 0.02 * base_ambiguity)
    add_bias_0 = rng.normal(0.0, 0.2 * base_ambiguity)
    spatial_bias_0 = (
        0.15
        * base_ambiguity
        * np.sin(4 * np.pi * lat_norm)
        * np.cos(4 * np.pi * lon_norm)
    )
    sigma_0 = 0.8 * base_ambiguity
    s0 = (
        scale_0 * y_true
        + add_bias_0
        + spatial_bias_0
        + rng.normal(0.0, sigma_0, n_samples)
    )

    sensors = [s0]

    # Sensor 1: moderate noise, heteroscedastic with y_true
    scale_1 = 1.2
    add_bias_1 = rng.normal(0.0, 0.5 * base_ambiguity)
    spatial_bias_1 = 0.4 * base_ambiguity * np.cos(6 * np.pi * lon_norm)
    sigma_1 = (1.0 + 0.02 * y_true) * base_ambiguity
    s1 = (
        scale_1 * y_true
        + add_bias_1
        + spatial_bias_1
        + rng.normal(0.0, sigma_1, n_samples)
    )
    sensors.append(s1)

    # Sensor 2: highest noise with occasional outliers
    scale_2 = 1.4
    add_bias_2 = rng.normal(0.0, 0.7 * base_ambiguity)
    spatial_bias_2 = 0.5 * base_ambiguity * np.sin(5 * np.pi * lat_norm * lon_norm)
    sigma_2 = (1.3 + 0.03 * y_true) * base_ambiguity
    outlier = rng.random(n_samples) < 0.05
    noise = rng.normal(0.0, sigma_2, n_samples)
    noise[outlier] += rng.normal(0.0, 3.0 * sigma_2[outlier])
    s2 = scale_2 * y_true + add_bias_2 + spatial_bias_2 + noise
    sensors.append(s2)

    B = np.column_stack(sensors)
    B = np.maximum(0.0, B)
    return B.astype(np.float32)


def generate_data(out_path):
    """Generate the fixed three-sensor Toy input used by the released tables."""
    n_samples, seed = 200000, 42
    lat_min, lat_max = 51.18, 51.25
    lon_min, lon_max = 4.35, 4.45
    ambiguity_level, spatial_noise_std = 3.0, 5.0
    n_sensors = 3
    print(
        f"Generating spatial data with {n_sensors} sensors, {n_samples} samples, "
        f"ambiguity={ambiguity_level}, spatial_noise={spatial_noise_std}..."
    )
    rng = np.random.default_rng(seed)

    # Generate and normalize coordinates
    lat = rng.uniform(lat_min, lat_max, n_samples)
    lon = rng.uniform(lon_min, lon_max, n_samples)
    lat_norm, lon_norm, _ = _normalize_coordinates(
        lat, lon, lat_min, lat_max, lon_min, lon_max
    )

    # Create covariates
    X = _create_spatial_features(lat_norm, lon_norm)
    x_cols = ["Latitude", "Longitude", "sin_lat", "cos_lat", "sin_lon", "cos_lon"]

    # Generate ground truth and sensor data
    y_true = _generate_ground_truth_field(lat_norm, lon_norm, rng, spatial_noise_std)
    B = _generate_sensors(y_true, lat_norm, lon_norm, rng, ambiguity_level)
    b_cols = ["sensor_0", "sensor_1", "sensor_2"]

    # Create DataFrame
    data_matrix = np.column_stack([lat, lon, X[:, 2:], B, y_true])
    all_cols = ["Latitude_raw", "Longitude_raw", *x_cols[2:], *b_cols, "y_true"]
    df = pd.DataFrame(data_matrix, columns=pd.Index(all_cols))

    _save_dataframe(df, out_path)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate the released spatial Toy dataset."
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/toy_spatial/toy_data.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    generate_data(args.out_path)


if __name__ == "__main__":
    main()
