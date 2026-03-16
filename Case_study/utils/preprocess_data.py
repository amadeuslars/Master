"""
Merge daily delivery CSVs into a unified customers.csv.

Usage:
    python Case_study/utils/preprocess_data.py

Reads all CSV files from Case_study/data/raw/, deduplicates by kundenr,
cross-references with existing 40-customer data for addresses/coordinates,
and writes Case_study/data/customers.csv.

Expected columns in daily CSVs:
    kundenr, tur-id, kundenavn, ppl freece, volum, antall leveranser,
    ppl, leveringsdag, leveringstid fra, leveringstid til
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

# --- Paths ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'customers.csv')

# Existing data for cross-referencing addresses/coordinates
OLD_CUSTOMERS_FILE = os.path.join(DATA_DIR, 'customers_old.csv')
OLD_GEOCODED_FILE = os.path.join(DATA_DIR, 'geocoded_addresses.csv')

# --- Column mapping (Norwegian → English) ---
COLUMN_MAP = {
    'kundenr': 'customer_id',
    'tur-id': 'tour_id',
    'kundenavn': 'customer_name',
    'hvorav ppl frys': 'ppl_freeze',
    'volum': 'volume_m3',
    'antall leveranser': '_drop',
    'ppl': 'ppl',
    'leveringsdag kunde': '_delivery_day_raw',
    'leveringstid kunde fra': 'tw_start',
    'leveringstid kunde til': 'tw_end',
}

# --- Day mapping (Norwegian → 3-letter code) ---
DAY_MAP = {
    'mandag': 'mon',
    'tirsdag': 'tue',
    'onsdag': 'wed',
    'torsdag': 'thu',
    'fredag': 'fri',
}


def read_raw_csvs(raw_dir):
    """Read all CSV files from the raw directory and concatenate."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    if not csv_files:
        print(f"ERROR: No CSV files found in {raw_dir}")
        print("Place your daily delivery CSVs (monday.csv, tuesday.csv, ...) in that directory.")
        sys.exit(1)

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        frames.append(df)
        print(f"  Read {os.path.basename(f)}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total rows across all files: {len(combined)}")
    return combined


def rename_columns(df):
    """Rename Norwegian columns to English, drop unused columns."""
    rename = {}
    for col in df.columns:
        mapped = COLUMN_MAP.get(col)
        if mapped and mapped != '_drop':
            rename[col] = mapped
    df = df.rename(columns=rename)

    # Drop columns marked for removal
    drop_cols = [c for c in df.columns if COLUMN_MAP.get(c) == '_drop']
    df = df.drop(columns=drop_cols, errors='ignore')

    return df


def normalize_delivery_day(raw_day):
    """Convert Norwegian day name to 3-letter code."""
    if pd.isna(raw_day):
        return ''
    return DAY_MAP.get(str(raw_day).strip().lower(), str(raw_day).strip().lower())


def normalize_rows(df):
    """
    Keep one row per customer per delivery day with per-day quantities.
    Deduplicates within the same (customer_id, day) if needed.
    """
    # Convert delivery day
    df['delivery_day'] = df['_delivery_day_raw'].apply(normalize_delivery_day)
    df = df.drop(columns=['_delivery_day_raw'], errors='ignore')

    # Deduplicate within same (customer_id, delivery_day)
    grouped = df.groupby(['customer_id', 'delivery_day'], sort=False)

    records = []
    for (cid, day), group in grouped:
        row = group.iloc[0].to_dict()
        row['delivery_day'] = day
        # For numeric fields, take first non-null within same day
        for col in ['ppl', 'ppl_freeze', 'volume_m3', 'tw_start', 'tw_end', 'tour_id']:
            vals = group[col].dropna()
            if len(vals) > 0:
                row[col] = vals.iloc[0]
        records.append(row)

    result = pd.DataFrame(records)
    unique_customers = result['customer_id'].nunique()
    print(f"  Rows (customer-day pairs): {len(result)}")
    print(f"  Unique customers: {unique_customers}")
    return result


def cross_reference_existing_data(df):
    """
    Match by customer_id (kundenr) against the old 40-customer dataset
    to pull address, postal_code, latitude, longitude.
    """
    # Try loading old customers file (renamed copy of original customers.csv)
    old_customers = None
    if os.path.exists(OLD_CUSTOMERS_FILE):
        old_customers = pd.read_csv(OLD_CUSTOMERS_FILE)
        old_customers.columns = old_customers.columns.str.strip()
        print(f"  Loaded old customers data: {len(old_customers)} rows")

    # Try loading geocoded addresses
    old_geocoded = None
    if os.path.exists(OLD_GEOCODED_FILE):
        old_geocoded = pd.read_csv(OLD_GEOCODED_FILE)
        print(f"  Loaded geocoded addresses: {len(old_geocoded)} rows")

    # Build address → (lat, lon) lookup from geocoded file
    geo_lookup = {}
    if old_geocoded is not None:
        for _, row in old_geocoded.iterrows():
            addr = str(row['Adresse']).strip()
            if addr != 'Depot' and pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                geo_lookup[addr] = (float(row['Latitude']), float(row['Longitude']))

    # Build kundenr → {address, postal_code, lat, lon} from old customers
    old_lookup = {}
    if old_customers is not None:
        for _, row in old_customers.iterrows():
            cid = int(row['Kundenr'])
            addr = str(row.get('Adresse', '')).strip()
            postnr = str(row.get('Postnr', '')).strip()
            # Get coordinates from geocoded lookup
            lat, lon = geo_lookup.get(addr, (np.nan, np.nan))
            old_lookup[cid] = {
                'address': addr if addr else '',
                'postal_code': postnr if postnr else '',
                'latitude': lat,
                'longitude': lon,
            }

    # Initialize new columns
    df['address'] = ''
    df['postal_code'] = ''
    df['latitude'] = np.nan
    df['longitude'] = np.nan
    df['geocode_status'] = 'missing'

    matched = 0
    for idx, row in df.iterrows():
        cid = int(row['customer_id'])
        if cid in old_lookup:
            old = old_lookup[cid]
            df.at[idx, 'address'] = old['address']
            df.at[idx, 'postal_code'] = old['postal_code']
            df.at[idx, 'latitude'] = old['latitude']
            df.at[idx, 'longitude'] = old['longitude']
            if pd.notna(old['latitude']) and pd.notna(old['longitude']):
                df.at[idx, 'geocode_status'] = 'found'
                matched += 1

    print(f"  Cross-referenced: {matched} customers matched with coordinates")
    return df


def format_time(val):
    """Ensure time values are in HH:MM format."""
    if pd.isna(val):
        return ''
    s = str(val).strip()
    # Already HH:MM
    if ':' in s and len(s) <= 5:
        return s
    # Try parsing as time
    try:
        t = pd.to_datetime(s, format='%H:%M')
        return t.strftime('%H:%M')
    except Exception:
        pass
    try:
        t = pd.to_datetime(s)
        return t.strftime('%H:%M')
    except Exception:
        return s


def build_output(df):
    """Assemble the final customers.csv with the target schema."""
    # Format time windows
    df['tw_start'] = df['tw_start'].apply(format_time)
    df['tw_end'] = df['tw_end'].apply(format_time)

    # Add default columns
    df['weight_kg'] = 0.0
    df['vehicle_type'] = 1
    df['service_time_min'] = 20.0
    df['notes'] = ''

    # Select and order columns
    output_cols = [
        'customer_id', 'customer_name', 'address', 'postal_code',
        'latitude', 'longitude', 'geocode_status', 'delivery_day',
        'tw_start', 'tw_end', 'ppl', 'ppl_freeze', 'volume_m3',
        'weight_kg', 'vehicle_type', 'tour_id', 'service_time_min', 'notes',
    ]

    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    return df[output_cols]


def main():
    print("=== Preprocessing: Merge daily CSVs → customers.csv ===\n")

    # 1. Read raw CSVs
    print("Step 1: Reading raw daily CSVs...")
    raw_df = read_raw_csvs(RAW_DIR)

    # 2. Rename columns
    print("\nStep 2: Renaming columns...")
    df = rename_columns(raw_df)
    print(f"  Columns: {list(df.columns)}")

    # 3. Normalize (one row per customer-day)
    print("\nStep 3: Normalizing to one row per customer per day...")
    df = normalize_rows(df)

    # 4. Cross-reference existing data
    print("\nStep 4: Cross-referencing with existing customer data...")
    df = cross_reference_existing_data(df)

    # 5. Build output
    print("\nStep 5: Building output...")
    output = build_output(df)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Wrote {len(output)} rows to {OUTPUT_FILE}")

    # Summary
    has_coords = output['geocode_status'].eq('found').sum()
    missing_coords = output['geocode_status'].ne('found').sum()
    print(f"\n=== Summary ===")
    print(f"  Total rows (customer-day): {len(output)}")
    print(f"  Unique customers: {output['customer_id'].nunique()}")
    print(f"  With coordinates: {has_coords}")
    print(f"  Missing coordinates: {missing_coords}")
    print(f"  Delivery day distribution:")
    for day in ['mon', 'tue', 'wed', 'thu', 'fri']:
        count = (output['delivery_day'] == day).sum()
        if count > 0:
            print(f"    {day}: {count} deliveries")


if __name__ == "__main__":
    main()
