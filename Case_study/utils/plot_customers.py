"""
Create an interactive HTML map with all customers from a CSV file.

Usage:
    python Case_study/utils/plot_customers.py
    python Case_study/utils/plot_customers.py --input Case_study/data/customers.csv --output Case_study/customer_map_all.html
    python Case_study/utils/plot_customers.py --unique-customers
"""

import argparse
import os

import folium
import pandas as pd


DEFAULT_INPUT = "Case_study/data/customers.csv"
DEFAULT_OUTPUT = "Case_study/customer_map_all.html"
DEPOT_LAT = 62.4293036
DEPOT_LON = 6.3280543


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HTML map of all customers")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input customers CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output HTML map")
    parser.add_argument(
        "--unique-customers",
        action="store_true",
        help="Keep only one marker per customer_id (first occurrence)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_cols = {"customer_id", "latitude", "longitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.unique_customers:
        df = df.drop_duplicates(subset=["customer_id"], keep="first").copy()

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        raise ValueError("No valid rows with latitude/longitude to map.")

    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

    folium.Marker(
        location=[DEPOT_LAT, DEPOT_LON],
        popup="<b>Depot</b><br>HI Giørtz",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for _, row in df.iterrows():
        cid = row.get("customer_id", "")
        cname = row.get("customer_name", "")
        addr = row.get("address", "")
        postal = row.get("postal_code", "")
        dday = row.get("delivery_day", "")

        popup_lines = [
            f"<b>{cname}</b>" if pd.notna(cname) and str(cname).strip() else "<b>Customer</b>",
            f"Customer ID: {cid}",
        ]
        if pd.notna(addr) and str(addr).strip():
            popup_lines.append(f"Address: {addr}")
        if pd.notna(postal) and str(postal).strip():
            popup_lines.append(f"Postal code: {postal}")
        if pd.notna(dday) and str(dday).strip():
            popup_lines.append(f"Delivery day: {dday}")

        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=4,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.7,
            popup="<br>".join(popup_lines),
        ).add_to(m)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    m.save(args.output)

    print(f"Saved HTML map: {args.output}")
    print(f"Mapped points: {len(df)}")


if __name__ == "__main__":
    main()
