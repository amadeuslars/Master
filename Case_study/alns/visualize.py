import folium
import math
import re
import pandas as pd
import requests
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feasibility import get_earliest_departure, compute_route_schedule

OSRM_SERVER_URL = "http://localhost:5001"

def _generate_route_colors(n):
    """Generate n visually distinct colors using HSL rotation."""
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360  # golden angle for good spread
        sat = 65 + (i % 3) * 10    # vary saturation slightly
        light = 45 + (i % 2) * 10  # vary lightness slightly
        # Convert HSL to hex
        import colorsys
        r, g, b = colorsys.hls_to_rgb(hue / 360, light / 100, sat / 100)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return colors

ROUTE_COLORS = _generate_route_colors(72)


def get_osrm_route_geometry(waypoints):
    """
    Query OSRM for the actual road geometry between a list of waypoints.

    Args:
        waypoints: list of (lat, lon) tuples

    Returns:
        list of [lat, lon] coordinates for the full road path, or None if OSRM fails
    """
    if len(waypoints) < 2:
        return None

    # OSRM expects lon,lat pairs separated by semicolons
    coords_str = ';'.join(f"{lon},{lat}" for lat, lon in waypoints)
    url = f"{OSRM_SERVER_URL}/route/v1/driving/{coords_str}?overview=full&geometries=geojson"

    resp = requests.get(url, timeout=10, proxies={'http': None, 'https': None})
    data = resp.json()
    geojson_coords = data['routes'][0]['geometry']['coordinates']
    return [[coord[1], coord[0]] for coord in geojson_coords]


def plot_solution(solution, customers_df, customer_arrays=None, output_file='route_map.html',
                  time_matrix_array=None, customer_addr_idx=None, depot_idx=0):
    """
    Plot ALNS solution routes on an interactive Folium map.

    Args:
        solution: Solution object with .routes, .vehicles, .route_meta
        customers_df: DataFrame with latitude/longitude columns (new or legacy schema)
        customer_arrays: dict with 'demand', 'tw_start', 'tw_end' arrays (optional, for richer popups)
        output_file: path to save the HTML map
        time_matrix_array: time matrix for schedule computation (optional)
        customer_addr_idx: address index array (optional)
        depot_idx: depot index in matrix (optional)
    """
    # Build geo_lookup from customers_df
    # Detect schema: new has 'latitude'/'longitude', legacy uses geocoded_addresses.csv
    geo_lookup = {}
    if 'latitude' in customers_df.columns:
        # New schema: lat/lon in customers_df
        for _, row in customers_df.iterrows():
            addr = str(row.get('address', '')).strip()
            if not addr or addr.lower() in ('nan', 'none'):
                addr = f"cust_{int(row['customer_id'])}"
            lat, lon = row.get('latitude'), row.get('longitude')
            if pd.notna(lat) and pd.notna(lon):
                geo_lookup[addr] = (float(lat), float(lon))
    else:
        # Legacy schema: load from geocoded_addresses.csv
        geocoded_path = 'Case_study/data/geocoded_addresses.csv'
        if os.path.exists(geocoded_path):
            geocoded_df = pd.read_csv(geocoded_path)
            for _, row in geocoded_df.iterrows():
                addr = str(row['Adresse']).strip()
                lat, lon = row['Latitude'], row['Longitude']
                if pd.notna(lat) and pd.notna(lon):
                    geo_lookup[addr] = (float(lat), float(lon))

    # Depot coordinates
    DEPOT_LAT, DEPOT_LON = 62.4293036, 6.3280543
    depot_coords = geo_lookup.get('Depot', (DEPOT_LAT, DEPOT_LON))

    m = folium.Map(location=list(depot_coords), zoom_start=13, tiles='OpenStreetMap')

    # Depot marker
    folium.Marker(
        location=list(depot_coords),
        popup='<b>Depot</b><br>HI Giørtz',
        icon=folium.Icon(color='black', icon='home', prefix='fa'),
    ).add_to(m)

    def fmt_time(h):
        hh = int(h)
        mm = int((h - hh) * 60)
        return f"{hh:02d}:{mm:02d}"

    # Add an empty overlay for "Hide all routes"
    fg_hide = folium.FeatureGroup(name="Hide all routes", show=True)
    fg_hide.add_to(m)

    # Pre-pass: track coordinate occurrences to offset overlapping markers
    coord_occurrence = {}

    # Store overlays to control their default visibility
    overlays = []
    active_route_count = 0
    for i, (route, vehicle) in enumerate(zip(solution.routes, solution.vehicles)):
        if vehicle == 'dummy' or not route:
            continue

        active_route_count += 1
        color = ROUTE_COLORS[active_route_count % len(ROUTE_COLORS)]
        meta = solution.route_meta[i] if hasattr(solution, 'route_meta') and solution.route_meta else None
        if meta:
            group_name = f"{vehicle} S{meta['shift']}T{meta['trip']} ({len(route)} stops)"
        else:
            group_name = f"{vehicle} ({len(route)} stops)"
        fg = folium.FeatureGroup(name=group_name, show=False)  # show=False means not ticked by default

        # Pre-compute schedule for this route if data available
        arrival_times = {}    # customer_idx -> arrival_time (hours)
        drive_times = {}      # customer_idx -> drive time from previous stop (minutes)
        service_times = {}    # customer_idx -> service time (minutes)
        wait_times = {}       # customer_idx -> wait time at stop (minutes)
        if time_matrix_array is not None and customer_addr_idx is not None and customer_arrays is not None:
            earliest_dep = get_earliest_departure(
                solution, i, time_matrix_array, customer_addr_idx,
                customer_arrays, depot_idx
            )
            lunch_pos = solution.lunch_breaks[i] if hasattr(solution, 'lunch_breaks') else None
            events = compute_route_schedule(
                route, time_matrix_array, customer_addr_idx,
                customer_arrays, depot_idx, earliest_dep, lunch_pos
            )
            # Compute drive times from matrix directly
            for s_idx, cust in enumerate(route):
                cust_addr = customer_addr_idx[cust - 1]
                if s_idx == 0:
                    prev_addr = depot_idx
                else:
                    prev_addr = customer_addr_idx[route[s_idx - 1] - 1]
                drive_times[cust] = time_matrix_array[prev_addr, cust_addr] * 60  # hours -> min
                service_times[cust] = float(customer_arrays['service_time'][cust - 1]) * 60  # hours -> min

            for ev in events:
                if ev['event'] == 'arrive' and ev['customer'] is not None:
                    arrival_times[ev['customer']] = ev['time']
                    # Extract wait time from event details
                    wait_match = re.search(r'wait (\d+)min', ev['details'])
                    if wait_match:
                        wait_times[ev['customer']] = int(wait_match.group(1))

        # Collect waypoints: depot -> customers -> depot
        waypoints = [depot_coords]

        for stop_num, c in enumerate(route, start=1):
            row = customers_df.iloc[c - 1]
            # Handle both new and legacy schema column names
            if 'address' in customers_df.columns:
                addr = str(row.get('address', '')).strip()
                if not addr or addr.lower() in ('nan', 'none'):
                    addr = f"cust_{int(row['customer_id'])}"
                name = row.get('customer_name', '')
                kundenr = row.get('customer_id', '')
            else:
                addr = str(row['Adresse']).strip()
                name = row.get('Kundenavn', '')
                kundenr = row.get('Kundenr', '')

            base_coords = geo_lookup.get(addr)
            if base_coords is None:
                continue

            # Use coordinate tuple as key so customers at same location
            # get offset even if their address strings differ
            coord_key = (round(base_coords[0], 7), round(base_coords[1], 7))
            idx = coord_occurrence.get(coord_key, 0)
            coord_occurrence[coord_key] = idx + 1
            if idx == 0:
                coords = base_coords
            else:
                angle = idx * (2 * math.pi / 8)
                delta = 0.00015  # ~15 metres offset for better visibility
                coords = (base_coords[0] + delta * math.cos(angle), base_coords[1] + delta * math.sin(angle))

            waypoints.append(base_coords)  # OSRM route uses real address, not offset

            popup_lines = [
                f"<b>{name}</b>",
                f"Kundenr: {kundenr}",
                f"Address: {addr}",
                f"Stop: {stop_num}/{len(route)}",
                f"Vehicle: {group_name}",
            ]
            if customer_arrays is not None:
                tw_s = customer_arrays['tw_start'][c - 1]
                tw_e = customer_arrays['tw_end'][c - 1]
                demand = customer_arrays['demand'][c - 1]
                popup_lines.append(f"TW: {fmt_time(tw_s)} - {fmt_time(tw_e)}")
                if c in arrival_times:
                    popup_lines.append(f"Arrives: {fmt_time(arrival_times[c])}")
                if c in drive_times:
                    from_label = "depot" if stop_num == 1 else f"stop {stop_num - 1}"
                    popup_lines.append(f"Drive from {from_label}: {drive_times[c]:.0f} min")
                if c in wait_times and wait_times[c] > 0:
                    popup_lines.append(f"Wait: {wait_times[c]:.0f} min")
                if c in service_times:
                    popup_lines.append(f"Service: {service_times[c]:.0f} min")
                popup_lines.append(f"Demand: {demand:.1f} PPL")

            popup_html = '<br>'.join(popup_lines)

            # Add a numbered marker for each stop
            folium.Marker(
                location=list(coords),
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; color: white; background: {color}; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 24px; border: 2px solid black;">{stop_num}</div>'
                ),
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{stop_num}. {name}",
            ).add_to(fg)

        waypoints.append(depot_coords)

        road_coords = get_osrm_route_geometry(waypoints)
        folium.PolyLine(road_coords, color=color, weight=4, opacity=0.8).add_to(fg)
        fg.add_to(m)
        overlays.append(fg)

    # Unassigned customers (dummy route) - shown in grey
    dummy_idx = len(solution.routes) - 1
    if solution.vehicles[dummy_idx] == 'dummy' and solution.routes[dummy_idx]:
        fg_unassigned = folium.FeatureGroup(name=f"Unassigned ({len(solution.routes[dummy_idx])})", show=False)
        for c in solution.routes[dummy_idx]:
            row = customers_df.iloc[c - 1]
            if 'address' in customers_df.columns:
                addr = str(row.get('address', '')).strip()
                if not addr or addr.lower() in ('nan', 'none'):
                    addr = f"cust_{int(row['customer_id'])}"
                name = row.get('customer_name', '')
                kundenr = row.get('customer_id', '')
            else:
                addr = str(row['Adresse']).strip()
                name = row.get('Kundenavn', '')
                kundenr = row.get('Kundenr', '')
            coords = geo_lookup.get(addr)
            if coords is None:
                continue

            popup_html = f"<b>{name}</b><br>Kundenr: {kundenr}<br>{addr}<br><i>UNASSIGNED</i>"

            folium.CircleMarker(
                location=list(coords),
                radius=7,
                color='gray',
                fill=True,
                fill_color='gray',
                fill_opacity=0.5,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"UNASSIGNED: {name}",
            ).add_to(fg_unassigned)

        fg_unassigned.add_to(m)
        overlays.append(fg_unassigned)

    # Add LayerControl with overlays not shown by default
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_file)
    print(f"Map saved to {output_file} ({active_route_count} routes, {sum(len(r) for r, v in zip(solution.routes, solution.vehicles) if v != 'dummy')} assigned customers)")
    return m


def plot_from_alns(solution, delivery_day='tue', customers_file='Case_study/data/customers.csv',
                   output_file='route_map.html'):
    """
    Convenience wrapper: loads all data files and plots the solution.
    Call this directly after run_alns() returns.
    Must be called from the project root directory (UiB/Master/).
    """
    import sys, os
    case_study_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if case_study_dir not in sys.path:
        sys.path.insert(0, case_study_dir)

    from utils.utils import load_vrp_data

    customers_df = pd.read_csv(customers_file)

    # Filter to delivery day if using new schema
    if 'delivery_day' in customers_df.columns:
        customers_df = customers_df[
            customers_df['delivery_day'].eq(delivery_day)
        ].reset_index(drop=True)

    _, _, _, time_matrix_array, _, depot_idx, addr_idx, customer_arrays = load_vrp_data(
        delivery_day=delivery_day, customers_file=customers_file)

    plot_solution(solution, customers_df, customer_arrays, output_file,
                  time_matrix_array=time_matrix_array, customer_addr_idx=addr_idx,
                  depot_idx=depot_idx)


if __name__ == '__main__':
    from alns import run_alns

    customers_file = 'Case_study/data/customers_alesund_sula_tue.csv'
    day = 'tue'
    output = 'Case_study/route_map_alns.html'

    print(f"Running ALNS (customers={customers_file}, day={day})...")
    solution = run_alns(delivery_day=day, customers_file=customers_file)
    print("\nGenerating route map...")
    plot_from_alns(solution, delivery_day=day, customers_file=customers_file,
                   output_file=output)
