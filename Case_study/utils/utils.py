import numpy as np
import pandas as pd
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cost import calculate_route_cost


class Solution:
    def __init__(self, routes, vehicles, unassigned=None):
        self.routes = routes  # List of lists (customer indices)
        self.vehicles = vehicles  # List of vehicle names
        self.unassigned = unassigned if unassigned is not None else []
        self.cost = 0.0
    
    def get_vehicle_counts(self):
        return Counter(self.vehicles)

    def copy(self):
        new = Solution([r[:] for r in self.routes], list(self.vehicles), list(self.unassigned) if hasattr(self, 'unassigned') else None)
        new.cost = self.cost
        return new



def load_vrp_data(customers_file='Case_study/data/customers.csv', vehicles_file='Case_study/data/vehicles.csv', time_matrix_file='Case_study/data/time_matrix.csv'):
    customers_df = pd.read_csv(customers_file)
    vehicles_df = pd.read_csv(vehicles_file)
    time_matrix_df = pd.read_csv(time_matrix_file, index_col=0)
    time_matrix_df = time_matrix_df.apply(pd.to_numeric, errors='coerce').fillna(0.0) / 3600.0

    num_customers = len(customers_df)

    # 1. Customers dict (for generate_initial_solution)
    customers_dict = {
        'customer_id': customers_df['Kundenr'].to_numpy(dtype=np.int32),
    }

    # 2. Vehicles dict: vehicle_name -> capacity dict
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

    # 3. Time matrix (numpy array)
    time_matrix_array = time_matrix_df.to_numpy(dtype=np.float64)

    # 4. Address-to-index mapping from time matrix columns
    address_to_idx = {col.strip(): i for i, col in enumerate(time_matrix_df.columns)}
    depot_idx = address_to_idx.get('Depot', len(time_matrix_df.columns) - 1)

    # 5. Customer address indices (0-based, like benchmark: addr_idx[c-1] for customer c)
    customer_addresses = customers_df['Adresse'].tolist()
    addr_idx = np.array([address_to_idx[addr.strip()] for addr in customer_addresses], dtype=np.intp)

    # 6. Customer arrays (0-based, like benchmark: arr[c-1] for customer c)
    def time_str_to_hours(t_str):
        try:
            return pd.to_timedelta(str(t_str) + ':00').total_seconds() / 3600.0
        except:
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

    return customers_dict, vehicles_dict, vehicle_names, time_matrix_array, depot_idx, addr_idx, customer_arrays

def evaluate_solution(solution, customer_addr_idx, time_matrix_array, depot_idx):
    total_cost = 0.0
    for i, route in enumerate(solution.routes):
        route_cost = calculate_route_cost(route, customer_addr_idx, time_matrix_array, depot_idx)
        total_cost += route_cost
        if solution.vehicles[i] == 'dummy':
            total_cost += 100 * len(route)
    solution.cost = total_cost
    return total_cost

def generate_initial_solution(customers_dict, vehicle_names=None):
    """
    Create a Solution with one empty route per real vehicle, and a dummy route containing all customers.
    Vehicle names are set according to the vehicle types provided in vehicle_names (list of str).
    """
    # Number of real vehicles (excluding dummy)
    if vehicle_names is None:
        raise ValueError("vehicle_names must be provided as a list of vehicle type names.")
    num_real_vehicles = len([v for v in vehicle_names if v != 'dummy'])
    # All customer indices (assuming 1-based, as in Solomon)
    num_customers = len(customers_dict[list(customers_dict.keys())[0]])
    all_customers = list(range(1, num_customers + 1))
    # Create empty routes for real vehicles, and one route for dummy vehicle with all customers
    routes = [[] for _ in range(num_real_vehicles)]
    routes.append(all_customers)
    # Vehicle names: all real vehicle types, then 'dummy'
    vehicles = [v for v in vehicle_names if v != 'dummy'] + ['dummy']
    return Solution(routes, vehicles)
