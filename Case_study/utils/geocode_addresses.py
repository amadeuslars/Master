"""
Geocode missing customer addresses in customers.csv.

Usage:
    python Case_study/utils/geocode_addresses.py

For customers WITH an address but no lat/lon: geocodes via Nominatim.
For customers WITHOUT an address: searches by business name.
Outputs a report of found vs still-missing customers.
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- Paths ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEFAULT_CUSTOMERS_FILE = os.path.join(BASE_DIR, 'data', 'customers.csv')

# Depot coordinates (HI Giørtz, Ålesund)
DEPOT_LAT = 62.4293036
DEPOT_LON = 6.3280543


def geocode_by_address(address, geocode_fn, postal_code=None):
    """Geocode using a street address. Tries multiple query strategies."""
    queries = []
    if postal_code and str(postal_code).strip() not in ('', 'nan', 'None'):
        queries.append(f"{address}, {str(postal_code).strip()}, Norway")
    queries.append(f"{address}, Norway")
    queries.append(f"{address}, Ålesund, Norway")

    for query in queries:
        try:
            location = geocode_fn(query)
            if location:
                # Validate: should be in western Norway region
                if 61.0 <= location.latitude <= 63.5 and 4.5 <= location.longitude <= 8.5:
                    return location.latitude, location.longitude, address
        except Exception as e:
            print(f"  ERROR geocoding '{query}': {e}")
    return None, None, None


def extract_place_from_name(name):
    """Try to extract a place/town name from the business name."""
    # Known place names that appear in business names
    places = [
        'Geiranger', 'Brattvåg', 'Fosnavåg', 'Langevåg', 'Sjøholt', 'Spjelkavik',
        'Ulsteinvik', 'Hareid', 'Volda', 'Ørsta', 'Sykkylven', 'Stranda', 'Stryn',
        'Vestnes', 'Valderøy', 'Godøy', 'Vigra', 'Skodje', 'Vatne', 'Tomrefjord',
        'Stordal', 'Valldal', 'Hellesylt', 'Nordfjordeid', 'Eid', 'Måløy',
        'Bryggja', 'Olden', 'Loen', 'Utvik', 'Selje', 'Raudeberg',
        'Ålesund', 'Moa', 'Ellingsøy', 'Digernes', 'Hovdebygda',
        'Grodås', 'Hornindal', 'Ikornnes', 'Ikornes', 'Larsnes', 'Gursken',
        'Sæbø', 'Åheim', 'Åram', 'Norddal', 'Rovde', 'Flatraket',
        'Stadlandet', 'Oldeide', 'Bellingen', 'Fiksdal', 'Lepsøy',
        'Hjelledalen', 'Vikane', 'Tresfjord', 'Vartdal', 'Fiskåbygd',
        'Søvik', 'Svingen', 'Straumane', 'Nørve', 'Oppstryn',
        'Hjørungavåg', 'Tennfjord',
    ]
    for place in places:
        if place.lower() in name.lower():
            return place
    return None


def geocode_by_name(name, geocode_fn):
    """Search by business name with multiple strategies."""
    # Clean name: remove numeric prefixes, codes, year suffixes
    clean = name.strip()

    # Strategy 1: Full name + Norway (broad)
    strategies = [
        f"{clean}, Norway",
        f"{clean}, Møre og Romsdal, Norway",
    ]

    # Strategy 2: If place name found, try "business, place, Norway"
    place = extract_place_from_name(clean)
    if place:
        # Put place-specific search first (more precise)
        strategies.insert(0, f"{clean}, {place}, Norway")

    for query in strategies:
        try:
            location = geocode_fn(query)
            if location:
                # Validate: should be in western Norway (lat ~61-63, lon ~4.5-8.5)
                if 61.0 <= location.latitude <= 63.0 and 4.5 <= location.longitude <= 8.5:
                    raw_addr = location.address.split(',')[0].strip() if location.address else ''
                    return location.latitude, location.longitude, raw_addr
        except Exception as e:
            print(f"  ERROR geocoding '{query}': {e}")

    return None, None, None


def validate_all_geocoded(df):
    """Check that all customers have coordinates. Returns list of missing."""
    missing = df[df['latitude'].isna() | df['longitude'].isna()]
    return missing


def main():
    parser = argparse.ArgumentParser(description='Geocode customer addresses')
    parser.add_argument('--file', default=DEFAULT_CUSTOMERS_FILE,
                        help='Path to customers CSV file')
    parser.add_argument('--redo', action='store_true',
                        help='Clear all existing lat/lon and re-geocode everything')
    args = parser.parse_args()

    customers_file = args.file
    print(f"=== Geocoding customer coordinates ===")
    print(f"    File: {customers_file}")
    print(f"    Redo all: {args.redo}\n")

    if not os.path.exists(customers_file):
        print(f"ERROR: {customers_file} not found.")
        sys.exit(1)

    df = pd.read_csv(customers_file)
    total = len(df)

    # If --redo, clear all existing coordinates
    if args.redo:
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        df['geocode_status'] = np.nan
        print(f"Cleared all existing coordinates ({total} rows)\n")

    # Find rows needing geocoding
    needs_geocoding = df[df['latitude'].isna() | df['longitude'].isna()]
    already_done = total - len(needs_geocoding)
    print(f"Total customers: {total}")
    print(f"Already geocoded: {already_done}")
    print(f"Need geocoding: {len(needs_geocoding)}\n")

    if len(needs_geocoding) == 0:
        print("All customers already have coordinates.")
        return

    # Set up geocoder
    geolocator = Nominatim(user_agent="master_thesis_vrp_giorts/1.0")
    geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    found_by_address = 0
    still_missing = []

    for idx in needs_geocoding.index:
        cid = int(df.at[idx, 'customer_id'])
        name = str(df.at[idx, 'customer_name'])
        address = str(df.at[idx, 'address']).strip()
        postal_code = df.at[idx, 'postal_code'] if 'postal_code' in df.columns else None

        lat, lon, resolved_addr = None, None, None

        # Geocode by address if we have one
        if address and address.lower() not in ('', 'nan', 'none'):
            lat, lon, resolved_addr = geocode_by_address(address, geocode_fn, postal_code)
            if lat is not None:
                found_by_address += 1
                df.at[idx, 'latitude'] = lat
                df.at[idx, 'longitude'] = lon
                df.at[idx, 'geocode_status'] = 'found'
                print(f"  [{cid}] {name}: found by address → ({lat:.6f}, {lon:.6f})")
                continue

        # No fallback — leave as None so user can fill in manually
        still_missing.append({'customer_id': cid, 'customer_name': name})
        df.at[idx, 'geocode_status'] = 'missing'
        print(f"  [{cid}] {name}: NOT FOUND (address geocode failed, no fallback)")

    # Save updated file
    df.to_csv(customers_file, index=False)
    print(f"\nUpdated {customers_file}")

    # Report
    print(f"\n=== Geocoding Report ===")
    print(f"  Found by address:  {found_by_address}")
    print(f"  Still missing:     {len(still_missing)}")

    if still_missing:
        print(f"\n  Missing customers (fill in manually):")
        for m in still_missing:
            print(f"    customer_id={m['customer_id']}, name={m['customer_name']}")

    # Final validation
    df_check = pd.read_csv(customers_file)
    missing_final = validate_all_geocoded(df_check)
    if len(missing_final) == 0:
        print(f"\n  All {total} customers now have coordinates.")
    else:
        print(f"\n  WARNING: {len(missing_final)} customers still lack coordinates.")
        print(f"  Fill in lat/lon manually in {customers_file}")


if __name__ == "__main__":
    main()
