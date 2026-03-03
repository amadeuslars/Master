import folium
import math
import pandas as pd
import requests
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OSRM_SERVER_URL = "http://localhost:5001"

# Colorblind-friendly, high-contrast colors for up to 15 routes
ROUTE_COLORS = [
    '#377eb8',  # Blue
    '#e41a1c',  # Red
    '#4daf4a',  # Green
    '#984ea3',  # Purple
    '#ff7f00',  # Orange
    '#ffff33',  # Yellow
    '#a65628',  # Brown
    '#f781bf',  # Pink
    '#999999',  # Grey
    '#66c2a5',  # Teal
    '#fc8d62',  # Salmon
    '#8da0cb',  # Light Blue
    '#e78ac3',  # Light Pink
    '#a6d854',  # Light Green
    '#ffd92f',  # Bright Yellow
]


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


def plot_solution(solution, customers_df, geocoded_df, customer_arrays=None, output_file='route_map.html'):
    """
    Plot ALNS solution routes on an interactive Folium map.

    Args:
        solution: Solution object with .routes and .vehicles
        customers_df: DataFrame from customers.csv
        geocoded_df: DataFrame from geocoded_addresses.csv (Adresse, Latitude, Longitude)
        customer_arrays: dict with 'demand', 'tw_start', 'tw_end' arrays (optional, for richer popups)
        output_file: path to save the HTML map
    """
    # Address -> (lat, lon) lookup
    geo_lookup = {}
    for _, row in geocoded_df.iterrows():
        addr = str(row['Adresse']).strip()
        lat, lon = row['Latitude'], row['Longitude']
        if pd.notna(lat) and pd.notna(lon):
            geo_lookup[addr] = (float(lat), float(lon))

    depot_coords = geo_lookup.get('Depot')
    if depot_coords is None:
        raise ValueError("Depot not found in geocoded_addresses.csv")

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

    # Pre-pass: track address occurrences to offset overlapping markers
    addr_occurrence = {}

    # Store overlays to control their default visibility
    overlays = []
    active_route_count = 0
    for i, (route, vehicle) in enumerate(zip(solution.routes, solution.vehicles)):
        if vehicle == 'dummy' or not route:
            continue

        active_route_count += 1
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        group_name = f"{vehicle} ({len(route)} stops)"
        fg = folium.FeatureGroup(name=group_name, show=False)  # show=False means not ticked by default

        # Collect waypoints: depot -> customers -> depot
        waypoints = [depot_coords]

        for stop_num, c in enumerate(route, start=1):
            row = customers_df.iloc[c - 1]
            addr = str(row['Adresse']).strip()
            base_coords = geo_lookup.get(addr)
            if base_coords is None:
                continue

            idx = addr_occurrence.get(addr, 0)
            addr_occurrence[addr] = idx + 1
            if idx == 0:
                coords = base_coords
            else:
                angle = idx * (2 * math.pi / 8)
                delta = 0.00008  # ~8 metres
                coords = (base_coords[0] + delta * math.cos(angle), base_coords[1] + delta * math.sin(angle))

            waypoints.append(base_coords)  # OSRM route uses real address, not offset

            name = row.get('Kundenavn', '')
            kundenr = row.get('Kundenr', '')
            popup_lines = [
                f"<b>{name}</b>",
                f"Kundenr: {kundenr}",
                f"Address: {addr}",
                f"Stop: {stop_num}/{len(route)}",
                f"Vehicle: {vehicle}",
            ]
            if customer_arrays is not None:
                tw_s = customer_arrays['tw_start'][c - 1]
                tw_e = customer_arrays['tw_end'][c - 1]
                demand = customer_arrays['demand'][c - 1]
                popup_lines.append(f"TW: {fmt_time(tw_s)} - {fmt_time(tw_e)}")
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
            addr = str(row['Adresse']).strip()
            coords = geo_lookup.get(addr)
            if coords is None:
                continue

            name = row.get('Kundenavn', '')
            kundenr = row.get('Kundenr', '')
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


def plot_from_alns(solution, output_file='route_map.html'):
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

    customers_df = pd.read_csv('Case_study/data/customers.csv')
    geocoded_df = pd.read_csv('Case_study/data/geocoded_addresses.csv')
    _, _, _, _, _, _, _, customer_arrays = load_vrp_data()

    plot_solution(solution, customers_df, geocoded_df, customer_arrays, output_file)


if __name__ == '__main__':
    from alns import run_alns

    print("Running ALNS...")
    solution = run_alns()
    print("\nGenerating route map...")
    plot_from_alns(solution, output_file='Case_study/route_map.html')
