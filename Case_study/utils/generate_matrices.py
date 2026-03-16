"""
Generate per-day OSRM time and distance matrices.

Usage:
    python Case_study/utils/generate_matrices.py              # all days
    python Case_study/utils/generate_matrices.py tue           # single day
    python Case_study/utils/generate_matrices.py mon wed fri   # specific days

Filters customers by delivery day, queries OSRM table API (with chunking
for >100 locations), and saves matrices to data/matrices/{day}/.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import math

# --- Configuration ---
OSRM_SERVER_URL = "http://localhost:5001"
OSRM_CHUNK_SIZE = 100  # OSRM table API limit per source/destination batch

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CUSTOMERS_FILE = os.path.join(BASE_DIR, 'data', 'customers.csv')
MATRICES_DIR = os.path.join(BASE_DIR, 'data', 'matrices')

# Depot
DEPOT_LAT = 62.4293036
DEPOT_LON = 6.3280543
DEPOT_LABEL = 'Depot'

ALL_DAYS = ['mon', 'tue', 'wed', 'thu', 'fri']


def get_day_customers(df, day):
    """Filter customers who deliver on a given day."""
    mask = df['delivery_day'].eq(day)
    return df[mask].copy()


def coord_label(lat, lon):
    """Generate a coordinate-based label for matrix rows/columns."""
    return f"{lat:.7f}_{lon:.7f}"


def build_location_list(day_df):
    """
    Build deduplicated list of (label, lat, lon) for matrix generation.
    Uses coordinate-based labels to avoid ambiguity when multiple
    customers share the same location.
    Depot is always index 0.
    """
    depot_label = coord_label(DEPOT_LAT, DEPOT_LON)
    locations = [(depot_label, DEPOT_LAT, DEPOT_LON)]
    seen_coords = {(round(DEPOT_LAT, 7), round(DEPOT_LON, 7))}

    for _, row in day_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or pd.isna(lon):
            continue
        coord_key = (round(lat, 7), round(lon, 7))
        if coord_key not in seen_coords:
            seen_coords.add(coord_key)
            locations.append((coord_label(lat, lon), lat, lon))

    return locations


def query_osrm_table(all_coords, sources_idx, destinations_idx):
    """
    Query OSRM table API for a subset of sources and destinations.

    Args:
        all_coords: list of (lon, lat) tuples for ALL locations
        sources_idx: list of indices into all_coords to use as sources
        destinations_idx: list of indices into all_coords to use as destinations

    Returns:
        (distances, durations) as 2D lists, or (None, None) on failure.
    """
    coords_str = ";".join(f"{lon},{lat}" for lon, lat in all_coords)
    sources_str = ";".join(str(i) for i in sources_idx)
    destinations_str = ";".join(str(i) for i in destinations_idx)

    url = (
        f"{OSRM_SERVER_URL}/table/v1/car/{coords_str}"
        f"?sources={sources_str}&destinations={destinations_str}"
        f"&annotations=distance,duration"
    )

    try:
        resp = requests.get(url, timeout=120, proxies={'http': None, 'https': None})
        resp.raise_for_status()
        data = resp.json()

        if data.get('code') != 'Ok':
            print(f"  OSRM error: {data.get('code')} - {data.get('message', '')}")
            return None, None

        return data.get('distances'), data.get('durations')
    except requests.exceptions.RequestException as e:
        print(f"  OSRM request failed: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"  JSON decode error: {e}")
        return None, None


def generate_matrix(locations):
    """
    Generate full distance and duration matrices using chunked OSRM queries.

    For N locations, sends all coordinates in every request but chunks the
    sources/destinations parameters to stay within OSRM limits.
    """
    n = len(locations)
    labels = [loc[0] for loc in locations]
    all_coords = [(loc[2], loc[1]) for loc in locations]  # (lon, lat) for OSRM

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    time_matrix = np.zeros((n, n), dtype=np.float64)

    # Chunk sources; destinations = all indices
    all_indices = list(range(n))
    src_chunks = [all_indices[i:i + OSRM_CHUNK_SIZE] for i in range(0, n, OSRM_CHUNK_SIZE)]
    dst_chunks = [all_indices[i:i + OSRM_CHUNK_SIZE] for i in range(0, n, OSRM_CHUNK_SIZE)]

    total_queries = len(src_chunks) * len(dst_chunks)
    query_num = 0

    for src_chunk in src_chunks:
        for dst_chunk in dst_chunks:
            query_num += 1
            print(f"    OSRM query {query_num}/{total_queries} "
                  f"(sources {src_chunk[0]}-{src_chunk[-1]}, "
                  f"destinations {dst_chunk[0]}-{dst_chunk[-1]})")

            distances, durations = query_osrm_table(all_coords, src_chunk, dst_chunk)
            if distances is None or durations is None:
                print(f"    FAILED — aborting matrix generation")
                return None, None, None

            # Fill into the full matrix
            for i_local, i_global in enumerate(src_chunk):
                for j_local, j_global in enumerate(dst_chunk):
                    dist_matrix[i_global, j_global] = distances[i_local][j_local]
                    time_matrix[i_global, j_global] = durations[i_local][j_local]

    dist_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    time_df = pd.DataFrame(time_matrix, index=labels, columns=labels)

    return dist_df, time_df, labels


def generate_for_day(day, df):
    """Generate and save matrices for a single delivery day."""
    print(f"\n--- Generating matrices for {day.upper()} ---")

    day_df = get_day_customers(df, day)
    print(f"  Customers on {day}: {len(day_df)}")

    # Check for missing coordinates
    missing = day_df[day_df['latitude'].isna() | day_df['longitude'].isna()]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} customers lack coordinates, excluding them")
        day_df = day_df.dropna(subset=['latitude', 'longitude'])

    if len(day_df) == 0:
        print(f"  No customers with coordinates for {day}, skipping")
        return

    # Build location list (depot + unique addresses)
    locations = build_location_list(day_df)
    print(f"  Unique locations (incl. depot): {len(locations)}")

    # Generate matrices
    dist_df, time_df, labels = generate_matrix(locations)
    if dist_df is None:
        print(f"  FAILED to generate matrices for {day}")
        return

    # Save
    day_dir = os.path.join(MATRICES_DIR, day)
    os.makedirs(day_dir, exist_ok=True)

    dist_path = os.path.join(day_dir, 'distance_matrix.csv')
    time_path = os.path.join(day_dir, 'time_matrix.csv')

    dist_df.to_csv(dist_path)
    time_df.to_csv(time_path)
    print(f"  Saved {dist_path} ({len(labels)}x{len(labels)})")
    print(f"  Saved {time_path} ({len(labels)}x{len(labels)})")

    # Sanity check: print min/max travel times
    time_vals = time_df.to_numpy()
    np.fill_diagonal(time_vals, np.nan)
    print(f"  Travel time range: {np.nanmin(time_vals):.0f}s - {np.nanmax(time_vals):.0f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate OSRM time/distance matrices')
    parser.add_argument('days', nargs='*', default=None, help='Days to generate (e.g. tue)')
    parser.add_argument('--file', default=CUSTOMERS_FILE, help='Path to customers CSV')
    args = parser.parse_args()

    customers_file = args.file
    if not os.path.exists(customers_file):
        print(f"ERROR: {customers_file} not found.")
        sys.exit(1)

    df = pd.read_csv(customers_file)

    # Determine which days to generate
    if args.days:
        days = [d.lower() for d in args.days]
        invalid = [d for d in days if d not in ALL_DAYS]
        if invalid:
            print(f"ERROR: Invalid days: {invalid}. Use: {ALL_DAYS}")
            sys.exit(1)
    else:
        days = ALL_DAYS

    print(f"=== Generating OSRM matrices for: {', '.join(days)} ===")
    print(f"Customers file: {customers_file}")
    print(f"OSRM server: {OSRM_SERVER_URL}")

    for day in days:
        generate_for_day(day, df)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
