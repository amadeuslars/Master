import pandas as pd
import requests
import json
import time

# --- Configuration ---
OSRM_SERVER_URL = "http://localhost:5001" # OSRM server is running on port 5001
GEOCODED_DATA_FILE = "geocoded_addresses.csv"
DISTANCE_MATRIX_FILE = "distance_matrix.csv"
TIME_MATRIX_FILE = "time_matrix.csv"

def get_osrm_matrix(coordinates):
    """
    Fetches distance and duration matrices from the OSRM table API.

    Args:
        coordinates (list of tuple): List of (latitude, longitude) pairs.

    Returns:
        tuple: (distances_matrix, durations_matrix) or (None, None) if request fails.
    """
    if not coordinates:
        return None, None

    # OSRM expects longitude,latitude pairs separated by semicolons
    # Make sure to handle potential None values for failed geocoding
    valid_coords = [(lon, lat) for lat, lon in coordinates if lat is not None and lon is not None]
    if not valid_coords:
        print("No valid coordinates to query OSRM.")
        return None, None

    coords_str = ";".join([f"{lon},{lat}" for lon, lat in valid_coords])
    url = f"{OSRM_SERVER_URL}/table/v1/car/{coords_str}?annotations=distance,duration"

    print(f"Querying OSRM URL: {url[:100]}...") # Print a truncated URL for debugging

    try:
        # Explicitly set proxies to None to bypass any system/environment proxy settings
        response = requests.get(url, timeout=120, proxies={'http': None, 'https': None}) 
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        distances = data.get("distances")
        durations = data.get("durations")

        if distances and durations:
            return distances, durations
        else:
            print(f"Error: OSRM response missing 'distances' or 'durations' key. Response: {data}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error querying OSRM server: {e}")
        print("Please ensure your OSRM Docker container is running and accessible on", OSRM_SERVER_URL)
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from OSRM: {e}")
        return None, None

if __name__ == "__main__":
    try:
        # 1. Load geocoded addresses
        print(f"Loading geocoded addresses from {GEOCODED_DATA_FILE}...")
        geo_df = pd.read_csv(GEOCODED_DATA_FILE)

        # Handle rows where geocoding failed (Latitude or Longitude is NaN)
        # We'll use a placeholder for these to maintain the matrix size,
        # but exclude them from the OSRM query itself.
        original_labels = geo_df['Adresse'].tolist()
        
        valid_geo_df = geo_df.dropna(subset=['Latitude', 'Longitude'])
        
        if valid_geo_df.empty:
            print("No valid geocoded addresses found to create a matrix. Exiting.")
            exit()

        # Prepare coordinates for OSRM
        # OSRM expects (longitude, latitude)
        # It's important to map the valid_geo_df to the original_labels correctly
        valid_coordinates = list(zip(valid_geo_df['Latitude'], valid_geo_df['Longitude']))
        valid_labels = valid_geo_df['Adresse'].tolist()

        # OSRM's table API can take up to 100 coordinates
        # If you have more, you might need to chunk your requests.
        if len(valid_coordinates) > 100:
             print(f"Warning: More than 100 coordinates ({len(valid_coordinates)}). OSRM's table API has a limit of 100. Consider chunking requests for larger datasets.")
             # For now, we'll just take the first 100 to demonstrate
             valid_coordinates = valid_coordinates[:100]
             valid_labels = valid_labels[:100]
             print(f"Processing only the first 100 valid locations for demonstration.")


        print(f"Found {len(valid_coordinates)} valid geocoded locations to query OSRM.")

        # 2. Get distance and duration matrices from OSRM
        print("Querying OSRM server for distance and duration matrices...")
        distances_raw, durations_raw = get_osrm_matrix(valid_coordinates)

        if distances_raw is None or durations_raw is None:
            print("Failed to retrieve matrices from OSRM. Please check your OSRM server logs if it's running.")
        else:
            # 3. Create pandas DataFrames from the OSRM results for *valid* locations
            distance_df_valid = pd.DataFrame(distances_raw, index=valid_labels, columns=valid_labels)
            time_df_valid = pd.DataFrame(durations_raw, index=valid_labels, columns=valid_labels)

            # Reconstruct full matrices, filling NaNs for original labels that failed geocoding
            full_distance_df = pd.DataFrame(index=original_labels, columns=original_labels, dtype=float)
            full_time_df = pd.DataFrame(index=original_labels, columns=original_labels, dtype=float)

            # Fill in the valid data
            full_distance_df.loc[valid_labels, valid_labels] = distance_df_valid
            full_time_df.loc[valid_labels, valid_labels] = time_df_valid

            # For locations that failed geocoding, distances/times to/from them will be NaN
            # You might want to fill these with a very large number (e.g., float('inf'))
            # or handle them in your VRP solver. For now, they remain NaN.

            # 4. Save matrices to CSV
            full_distance_df.to_csv(DISTANCE_MATRIX_FILE)
            full_time_df.to_csv(TIME_MATRIX_FILE)

            print(f"\nSuccessfully created and saved distance matrix to {DISTANCE_MATRIX_FILE}")
            print(f"Successfully created and saved time matrix to {TIME_MATRIX_FILE}")
            print("\nSample Distance Matrix (first 5x5):")
            print(full_distance_df.head(5).iloc[:, :5])
            print("\nSample Time Matrix (first 5x5):")
            print(full_time_df.head(5).iloc[:, :5])

    except FileNotFoundError:
        print(f"Error: {GEOCODED_DATA_FILE} not found. Please run read_data.py first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")