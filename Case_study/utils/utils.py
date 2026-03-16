import numpy as np
import pandas as pd
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cost import calculate_route_cost

# Multi-trip constants
SHIFT_WINDOWS = {1: (6.0, 14.0), 2: (15.0, 23.0)}


class Solution:
    def __init__(self, routes, vehicles, unassigned=None, route_meta=None):
        self.routes = routes  # List of lists (customer indices)
        self.vehicles = vehicles  # List of vehicle names
        self.cost = 0.0
        self.lunch_breaks = [None] * len(routes)  # Per-route: position (1..n) after which lunch is taken
        self.route_meta = route_meta or [None] * len(routes)

    def get_vehicle_counts(self):
        return Counter(self.vehicles)

    def copy(self):
        new = Solution(
            [r[:] for r in self.routes],
            list(self.vehicles),
            route_meta=[dict(m) if m else None for m in self.route_meta],
        )
        new.cost = self.cost
        new.lunch_breaks = list(self.lunch_breaks)
        return new



def _detect_schema(customers_df):
    """Detect whether customers.csv uses old (Norwegian) or new (English) column names."""
    return 'customer_id' in customers_df.columns


def load_vrp_data(delivery_day='tue',
                  customers_file='Case_study/data/customers.csv',
                  vehicles_file='Case_study/data/vehicles.csv'):
    """
    Load VRP data for a specific delivery day.

    Supports both the new schema (customer_id, ppl, ...) and the legacy schema
    (Kundenr, PPL pr leveranse, ...) for backward compatibility.

    Returns: (customers_dict, vehicles_dict, vehicle_names, time_matrix_array,
              distance_matrix_array, depot_idx, addr_idx, customer_arrays)
    """
    customers_df = pd.read_csv(customers_file)
    vehicles_df = pd.read_csv(vehicles_file)
    new_schema = _detect_schema(customers_df)

    if new_schema:
        return _load_new_schema(customers_df, vehicles_df, delivery_day)
    else:
        return _load_legacy_schema(customers_df, vehicles_df)


def _load_new_schema(customers_df, vehicles_df, delivery_day):
    """Load from the new unified customers.csv with per-day matrices."""
    # Filter customers by delivery day
    customers_df = customers_df[
        customers_df['delivery_day'].eq(delivery_day)
    ].reset_index(drop=True)

    num_customers = len(customers_df)
    if num_customers == 0:
        raise ValueError(f"No customers found for delivery day '{delivery_day}'")

    # Load per-day matrices
    matrix_dir = f'Case_study/data/matrices/{delivery_day}'
    time_matrix_file = os.path.join(matrix_dir, 'time_matrix.csv')
    dist_matrix_file = os.path.join(matrix_dir, 'distance_matrix.csv')

    time_matrix_df = pd.read_csv(time_matrix_file, index_col=0)
    time_matrix_df = time_matrix_df.apply(pd.to_numeric, errors='coerce').fillna(0.0) / 3600.0
    dist_matrix_df = pd.read_csv(dist_matrix_file, index_col=0)
    dist_matrix_df = dist_matrix_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    time_matrix_array = time_matrix_df.to_numpy(dtype=np.float64)
    distance_matrix_array = dist_matrix_df.to_numpy(dtype=np.float64)

    # 1. Customers dict
    customers_dict = {
        'customer_id': customers_df['customer_id'].to_numpy(dtype=np.int32),
    }

    # 2. Vehicles dict
    vehicles_dict = {}
    vehicle_names = []
    for _, row in vehicles_df.iterrows():
        name = row['Navn'].strip()
        vehicles_dict[name] = {
            'PPL total': float(row['PPL total']),
            'PPL Frys': float(row['PPL Frys']),
            'm3': float(row['m3']),
            'Vekt (KG)': float(row['Vekt (KG)']),
        }
        vehicle_names.extend([name] * int(row['antall']))

    # 3. Coordinate-based index mapping from time matrix columns
    col_to_idx = {col.strip(): i for i, col in enumerate(time_matrix_df.columns)}

    # Depot: find by coordinate label
    depot_coord_label = f"{62.4293036:.7f}_{6.3280543:.7f}"
    depot_idx = col_to_idx.get(depot_coord_label, 0)

    # 4. Customer coordinate indices (with fuzzy matching for small coord diffs)
    def get_coord_key(row):
        return f"{row['latitude']:.7f}_{row['longitude']:.7f}"

    # Build reverse lookup: parse matrix column labels into (lat, lon) floats
    matrix_coords = []
    for col in time_matrix_df.columns:
        col = col.strip()
        parts = col.split('_')
        if len(parts) == 2:
            try:
                matrix_coords.append((float(parts[0]), float(parts[1]), col))
            except ValueError:
                pass

    def find_closest_col(lat, lon, threshold=0.001):
        """Find matrix column closest to (lat, lon). Threshold in degrees (~100m)."""
        exact_key = f"{lat:.7f}_{lon:.7f}"
        if exact_key in col_to_idx:
            return exact_key
        best_dist = float('inf')
        best_col = None
        for mlat, mlon, mcol in matrix_coords:
            d = (mlat - lat)**2 + (mlon - lon)**2
            if d < best_dist:
                best_dist = d
                best_col = mcol
        if best_dist < threshold**2:
            return best_col
        return None

    addr_indices = []
    missing_customers = []
    for i, row in customers_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or pd.isna(lon):
            missing_customers.append((i, int(row['customer_id']), str(row.get('customer_name', ''))))
            addr_indices.append(0)  # placeholder — will be excluded
            continue
        col_key = find_closest_col(lat, lon)
        if col_key is None:
            missing_customers.append((i, int(row['customer_id']), str(row.get('customer_name', ''))))
            addr_indices.append(0)  # placeholder
        else:
            addr_indices.append(col_to_idx[col_key])

    if missing_customers:
        print(f"WARNING: {len(missing_customers)} customers not found in time matrix (need matrix regeneration):")
        for idx, cid, name in missing_customers:
            print(f"  customer_id={cid}, name={name}")

    addr_idx = np.array(addr_indices, dtype=np.intp)

    # 5. Customer arrays
    def time_str_to_hours(t_str):
        if pd.isna(t_str) or str(t_str).strip() == '':
            return 0.0
        try:
            parts = str(t_str).strip().split(':')
            return int(parts[0]) + int(parts[1]) / 60.0
        except Exception:
            return 0.0

    service_minutes = customers_df['service_time_min'].fillna(20.0).to_numpy(np.float64)

    tw_start_arr = customers_df['tw_start'].apply(time_str_to_hours).to_numpy(np.float64)
    tw_end_arr = customers_df['tw_end'].apply(time_str_to_hours).to_numpy(np.float64)

    # Clamp time windows that barely overshoot shift-1 end:
    # Only clamp if tw_end extends past shift-1 end by a small margin (<= 30 min)
    # AND the customer's TW doesn't reach into shift-2 territory.
    # Customers with TWs spanning the gap (e.g. 13:15-15:15) are left unclamped
    # so the repair operators can assign them to either shift.
    shift1_end = SHIFT_WINDOWS[1][1]
    shift2_start = SHIFT_WINDOWS[2][0]
    CLAMP_MARGIN = 0.5  # Only clamp if tw_end overshoots by <= 30 min
    for i in range(len(tw_start_arr)):
        if (tw_start_arr[i] < shift1_end
                and tw_end_arr[i] > shift1_end
                and tw_end_arr[i] <= shift1_end + CLAMP_MARGIN):
            tw_end_arr[i] = shift1_end

    customer_arrays = {
        'demand': customers_df['ppl'].fillna(0.0).to_numpy(np.float64),
        'tw_start': tw_start_arr,
        'tw_end': tw_end_arr,
        'service_time': service_minutes / 60.0,  # convert to hours
        'pallets': customers_df['ppl'].fillna(0.0).to_numpy(np.float64),
        'frys': customers_df['ppl_freeze'].fillna(0.0).to_numpy(np.float64),
        'volume_m3': customers_df['volume_m3'].fillna(0.0).to_numpy(np.float64),
        'weight_kg': customers_df['weight_kg'].fillna(0.0).to_numpy(np.float64),
        'biltype': customers_df['vehicle_type'].fillna(1).to_numpy(np.int32),
    }

    return customers_dict, vehicles_dict, vehicle_names, time_matrix_array, distance_matrix_array, depot_idx, addr_idx, customer_arrays


def _load_legacy_schema(customers_df, vehicles_df):
    """Load from the old Norwegian-column customers.csv with flat matrices."""
    time_matrix_file = 'Case_study/data/time_matrix.csv'
    dist_matrix_file = 'Case_study/data/distance_matrix.csv'

    time_matrix_df = pd.read_csv(time_matrix_file, index_col=0)
    time_matrix_df = time_matrix_df.apply(pd.to_numeric, errors='coerce').fillna(0.0) / 3600.0
    dist_matrix_df = pd.read_csv(dist_matrix_file, index_col=0)
    dist_matrix_df = dist_matrix_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    distance_matrix_array = dist_matrix_df.to_numpy(dtype=np.float64)

    num_customers = len(customers_df)

    customers_dict = {
        'customer_id': customers_df['Kundenr'].to_numpy(dtype=np.int32),
    }

    vehicles_dict = {}
    vehicle_names = []
    for _, row in vehicles_df.iterrows():
        name = row['Navn'].strip()
        vehicles_dict[name] = {
            'PPL total': float(row['PPL total']),
            'PPL Frys': float(row['PPL Frys']),
            'm3': float(row['m3']),
            'Vekt (KG)': float(row['Vekt (KG)']),
        }
        vehicle_names.extend([name] * int(row['antall']))

    time_matrix_array = time_matrix_df.to_numpy(dtype=np.float64)

    address_to_idx = {col.strip(): i for i, col in enumerate(time_matrix_df.columns)}
    depot_idx = address_to_idx.get('Depot', len(time_matrix_df.columns) - 1)

    customer_addresses = customers_df['Adresse'].tolist()
    addr_idx = np.array([address_to_idx[addr.strip()] for addr in customer_addresses], dtype=np.intp)

    def time_str_to_hours(t_str):
        try:
            return pd.to_timedelta(str(t_str) + ':00').total_seconds() / 3600.0
        except Exception:
            return 0.0

    biltype_raw = customers_df['Biltype .1'].fillna(1).to_numpy(np.int32) if 'Biltype .1' in customers_df.columns else customers_df['Biltype'].fillna(1).to_numpy(np.int32)

    customer_arrays = {
        'demand': customers_df['PPL pr leveranse'].fillna(0.0).to_numpy(np.float64),
        'tw_start': customers_df['Leveringstid kunde fra'].apply(time_str_to_hours).fillna(0.0).to_numpy(np.float64),
        'tw_end': customers_df['Leveringstid kunde til'].apply(time_str_to_hours).fillna(24.0).to_numpy(np.float64),
        'service_time': np.full(num_customers, 0.5, dtype=np.float64),
        'pallets': customers_df['PPL pr leveranse'].fillna(0.0).to_numpy(np.float64),
        'frys': customers_df['PPL hvorav frys'].fillna(0.0).to_numpy(np.float64),
        'volume_m3': customers_df['Volum pr leveranse (m3)'].fillna(0.0).to_numpy(np.float64),
        'weight_kg': customers_df['Vekt pr leveranse (tonn)'].fillna(0.0).to_numpy(np.float64) * 1000,
        'biltype': biltype_raw,
    }

    return customers_dict, vehicles_dict, vehicle_names, time_matrix_array, distance_matrix_array, depot_idx, addr_idx, customer_arrays

def evaluate_solution(solution, customer_addr_idx, time_matrix_array, depot_idx):
    total_cost = 0.0
    for i, route in enumerate(solution.routes):
        if solution.vehicles[i] == 'dummy':
            total_cost += 100 * len(route)
        else:
            route_cost = calculate_route_cost(route, customer_addr_idx, time_matrix_array, depot_idx)
            total_cost += route_cost
    solution.cost = total_cost
    return total_cost

def generate_initial_solution(customers_dict, vehicle_names=None):
    """
    Create a Solution with 4 route slots per vehicle (2 shifts x 2 trips)
    plus one dummy route containing all customers.
    Route order per vehicle: S1T1, S1T2, S2T1, S2T2.
    """
    if vehicle_names is None:
        raise ValueError("vehicle_names must be provided as a list of vehicle type names.")

    real_names = [v for v in vehicle_names if v != 'dummy']
    num_customers = len(customers_dict[list(customers_dict.keys())[0]])
    all_customers = list(range(1, num_customers + 1))

    routes = []
    vehicles = []
    route_meta = []

    for v_idx, v_name in enumerate(real_names):
        for shift in [1, 2]:
            shift_start, shift_end = SHIFT_WINDOWS[shift]
            for trip in [1, 2]:
                routes.append([])
                vehicles.append(v_name)
                route_meta.append({
                    'vehicle_idx': v_idx,
                    'shift': shift,
                    'trip': trip,
                    'shift_start': shift_start,
                    'shift_end': shift_end,
                })

    # Dummy route (last)
    routes.append(all_customers)
    vehicles.append('dummy')
    route_meta.append(None)

    return Solution(routes, vehicles, route_meta=route_meta)
