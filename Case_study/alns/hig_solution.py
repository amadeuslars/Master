"""
Build HIG's current solution from tour_id assignments for comparison.
Routes are sorted by tw_start (earliest first) within each tour.
"""
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_vrp_data, Solution, evaluate_solution, WORK_START
from utils.feasibility import (
    get_earliest_departure, compute_route_schedule, compute_trip_return_time,
    check_time_window_feasibility, check_capacity_feasibility,
    check_lunch_break_feasibility, LOADING_TIME, DELOADING_TIME, LUNCH_DURATION
)
from alns import _fmt_time


def build_hig_solution(delivery_day='mon', customers_file='Case_study/data/training_instances/real_tue.csv',
                       subset_tour_ids=None):
    """
    Build a Solution object from HIG's tour_id groupings.

    Args:
        delivery_day: which day to load
        customers_file: path to customers CSV
        subset_tour_ids: list of tour_ids to include (None = all)

    Returns:
        solution, customers_df, time_matrix_array, addr_idx, customer_arrays, depot_idx
    """
    customers_dict, vehicles_dict, vehicle_names, time_matrix_array, _, depot_idx, addr_idx, customer_arrays = load_vrp_data(delivery_day=delivery_day, customers_file=customers_file)

    # Load customer data for tour_id mapping
    full_df = pd.read_csv(customers_file)
    if 'delivery_day' in full_df.columns:
        day_df = full_df[full_df['delivery_day'] == delivery_day].reset_index(drop=True)
    else:
        day_df = full_df.reset_index(drop=True)

    # Map customer_id -> 1-based index in day dataset
    cid_to_idx = {int(row['customer_id']): i + 1 for i, row in day_df.iterrows()}

    # Filter to subset if provided
    if subset_tour_ids is not None:
        day_df = day_df[day_df['tour_id'].isin(subset_tour_ids)]

    # Group by tour_id, sort each group by tw_start
    tour_ids = sorted(day_df['tour_id'].unique())
    routes = []
    vehicles = []
    route_meta = []

    for tid in tour_ids:
        grp = day_df[day_df['tour_id'] == tid].sort_values('tw_start')
        route = [cid_to_idx[int(row['customer_id'])] for _, row in grp.iterrows()]
        routes.append(route)
        vehicles.append(f'tour-{int(tid)}')
        route_meta.append({
            'vehicle_idx': len(routes) - 1,
            'shift': 1,
            'trip': 1,
            'shift_start': 6.0,
            'shift_end': 22.0,  # generous — HIG doesn't have strict shift model
        })

    # Dummy route for customers not in subset
    all_assigned = set()
    for r in routes:
        all_assigned.update(r)
    num_customers = len(customers_dict['customer_id'])
    unassigned = [c for c in range(1, num_customers + 1) if c not in all_assigned]
    routes.append(unassigned)
    vehicles.append('dummy')
    route_meta.append(None)

    sol = Solution(routes, vehicles, route_meta=route_meta)
    sol.lunch_breaks = [None] * len(routes)
    cost = evaluate_solution(sol, addr_idx, time_matrix_array, depot_idx)

    return sol, day_df, time_matrix_array, addr_idx, customer_arrays, depot_idx, customers_dict, vehicles_dict


def print_hig_schedule(sol, customers_dict, time_matrix_array, addr_idx, customer_arrays, depot_idx):
    """Print schedule for HIG's solution."""
    cids = customers_dict['customer_id']
    num_assigned = sum(len(r) for r, v in zip(sol.routes, sol.vehicles) if v != 'dummy')
    num_unassigned = len(sol.routes[-1])

    print(f"\n--- HIG Current Solution ---")
    print(f"Total travel time: {sol.cost:.2f} hours ({sol.cost * 60:.0f} min)")
    print(f"Routes: {sum(1 for r, v in zip(sol.routes, sol.vehicles) if r and v != 'dummy')}")
    print(f"Assigned: {num_assigned} | Unassigned: {num_unassigned}")

    for i, (route, veh) in enumerate(zip(sol.routes, sol.vehicles)):
        if not route or veh == 'dummy':
            continue

        # Compute schedule with earliest possible departure (back-calculated from first TW)
        first_cust = route[0]
        first_tw = float(customer_arrays['tw_start'][first_cust - 1])
        first_addr = addr_idx[first_cust - 1]
        travel_to_first = time_matrix_array[depot_idx, first_addr]
        # Depart so driver arrives at first TW start (loading time = 1h)
        earliest_dep = max(0.0, first_tw - travel_to_first - 1.0)

        events = compute_route_schedule(
            route, time_matrix_array, addr_idx,
            customer_arrays, depot_idx, earliest_dep, lunch_pos=None
        )

        print(f"\n  === {veh} ({len(route)} stops) ===")
        for ev in events:
            cust_label = ""
            if ev['customer'] is not None:
                cust_label = f" [#{int(cids[ev['customer'] - 1])}]"
            print(f"    {_fmt_time(ev['time'])}  {ev['details']}{cust_label}")

        if events:
            drive_time = sol.cost  # individual route cost not tracked, show total at top


def check_hig_feasibility(sol, customers_dict, vehicles_dict, time_matrix_array,
                          addr_idx, customer_arrays, depot_idx):
    """
    Check HIG solution for feasibility violations.
    Since HIG routes use tour-XXXX names (not real vehicle types), capacity is
    checked against the largest vehicle (33 PPL) as an upper bound.
    """
    cids = customers_dict['customer_id']
    # Use largest vehicle for capacity checks (upper bound)
    largest_vehicle = 'large'

    violations = []

    for i, (route, veh) in enumerate(zip(sol.routes, sol.vehicles)):
        if not route or veh == 'dummy':
            continue

        # --- 1. Time window violations ---
        # Back-calculate departure like print_hig_schedule does
        first_cust = route[0]
        first_tw = float(customer_arrays['tw_start'][first_cust - 1])
        first_addr = addr_idx[first_cust - 1]
        travel_to_first = time_matrix_array[depot_idx, first_addr]
        earliest_dep = max(0.0, first_tw - travel_to_first - LOADING_TIME)

        # Simulate schedule to find specific TW violations
        depart_time = earliest_dep + LOADING_TIME
        current_time = depart_time
        last_addr = depot_idx

        for j, cust in enumerate(route):
            curr_addr = addr_idx[cust - 1]
            travel = time_matrix_array[last_addr, curr_addr]
            arrival = current_time + travel
            tw_start = float(customer_arrays['tw_start'][cust - 1])
            tw_end = float(customer_arrays['tw_end'][cust - 1])
            service = float(customer_arrays['service_time'][cust - 1])

            if arrival > tw_end + 0.01:
                violations.append({
                    'route': veh,
                    'type': 'TIME_WINDOW',
                    'detail': f'Customer #{int(cids[cust-1])} (stop {j+1}): '
                              f'arrives {arrival:.2f}h but tw_end={tw_end:.2f}h '
                              f'(late by {(arrival-tw_end)*60:.0f} min)'
                })

            current_time = max(arrival, tw_start) + service
            last_addr = curr_addr

        # --- 2. Capacity violations (against largest vehicle) ---
        if not check_capacity_feasibility(route, largest_vehicle, vehicles_dict, customer_arrays):
            idx = np.array([c - 1 for c in route], dtype=np.int32)
            total_ppl = float(customer_arrays['demand'][idx].sum())
            total_frys = float(customer_arrays['frys'][idx].sum())
            total_vol = float(customer_arrays['volume_m3'][idx].sum())
            total_wt = float(customer_arrays['weight_kg'][idx].sum())
            cap = vehicles_dict[largest_vehicle]
            details = []
            ppl_tol = 1/3 - 1e-6
            if total_ppl > cap['PPL total'] + ppl_tol:
                details.append(f'PPL {total_ppl:.2f}/{cap["PPL total"]:.1f}')
            if total_frys > cap['PPL Frys'] + ppl_tol:
                details.append(f'Frys {total_frys:.2f}/{cap["PPL Frys"]:.1f}')
            if total_vol > cap['m3']:
                details.append(f'Vol {total_vol:.0f}/{cap["m3"]:.0f}')
            if total_wt > cap['Vekt (KG)']:
                details.append(f'Weight {total_wt:.0f}/{cap["Vekt (KG)"]:.0f}')
            violations.append({
                'route': veh,
                'type': 'CAPACITY',
                'detail': f'Exceeds largest vehicle ({largest_vehicle}): {", ".join(details)}'
            })

        # --- 3. Route duration (load + deliver + return + deload + lunch) ---
        return_time = compute_trip_return_time(
            route, time_matrix_array, addr_idx, customer_arrays,
            depot_idx, earliest_dep
        )
        # Total includes loading, route, deloading, and lunch (part of the 8h shift)
        total_duration = return_time - earliest_dep + DELOADING_TIME + LUNCH_DURATION
        shift_duration = 8.0  # standard shift length (includes lunch)
        if total_duration > shift_duration + 0.01:
            violations.append({
                'route': veh,
                'type': 'SHIFT_DURATION',
                'detail': f'Total duration {total_duration:.2f}h '
                          f'({total_duration*60:.0f} min) exceeds {shift_duration}h shift '
                          f'(over by {(total_duration-shift_duration)*60:.0f} min)'
            })

        # --- 4. Lunch break feasibility ---
        # All trips start from the same work start
        shift_start = WORK_START

        if not check_lunch_break_feasibility(
            route, time_matrix_array, addr_idx, customer_arrays, depot_idx,
            lunch_duration=LUNCH_DURATION, earliest_departure=earliest_dep,
            shift_start=shift_start
        ):
            violations.append({
                'route': veh,
                'type': 'LUNCH_BREAK',
                'detail': f'No feasible lunch break position within first 6h of workday '
                          f'(shift_start={shift_start:.0f}h, departs {earliest_dep + LOADING_TIME:.2f}h)'
            })

    return violations


if __name__ == '__main__':
    # --- Configure here ---
    customers_file = 'Case_study/data/training_instances/real_tue.csv'
    delivery_day = 'tue'
    output_map = 'Case_study/hig_route_map.html'
    # ----------------------

    df_tours = pd.read_csv(customers_file)
    if 'delivery_day' in df_tours.columns:
        df_tours = df_tours[df_tours['delivery_day'] == delivery_day]
    all_tour_ids = sorted(df_tours['tour_id'].dropna().unique().tolist())
    print(f"Tour IDs found: {all_tour_ids}")

    sol, day_df, tm, addr_idx, ca, depot_idx, cust_dict, veh_dict = build_hig_solution(
        delivery_day=delivery_day, customers_file=customers_file, subset_tour_ids=all_tour_ids
    )

    print_hig_schedule(sol, cust_dict, tm, addr_idx, ca, depot_idx)

    viols = check_hig_feasibility(sol, cust_dict, veh_dict, tm, addr_idx, ca, depot_idx)
    if viols:
        print(f"\n--- Feasibility Violations ({len(viols)}) ---")
        for v in viols:
            print(f"  [{v['type']}] {v['route']}: {v['detail']}")
    else:
        print("\n--- No feasibility violations found ---")

    from visualize import plot_solution
    customers_df = pd.read_csv(customers_file)
    if 'delivery_day' in customers_df.columns:
        customers_df = customers_df[customers_df['delivery_day'] == delivery_day].reset_index(drop=True)
    else:
        customers_df = customers_df.reset_index(drop=True)
    plot_solution(sol, customers_df, ca, output_map,
                  time_matrix_array=tm, customer_addr_idx=addr_idx, depot_idx=depot_idx)
