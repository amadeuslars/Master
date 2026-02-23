import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# --- Geocoding Function ---
def geocode_addresses(addresses):
    """
    Converts a list of addresses into geographic coordinates (latitude, longitude).
    """
    geolocator = Nominatim(user_agent="master_thesis_vrp_gcli/1.0")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    print("Geocoding addresses... (This may take a moment)")
    
    results = []
    for address in addresses:
        try:
            full_address = f"{address}, Ålesund, Norway"
            location = geocode(full_address)
            if location:
                results.append((address, location.latitude, location.longitude))
                print(f"SUCCESS: {address} -> ({location.latitude}, {location.longitude})")
            else:
                results.append((address, None, None))
                print(f"FAILED:  {address} -> Not found")
        except Exception as e:
            print(f"ERROR:   {address} -> {e}")
            results.append((address, None, None))
        
    return results

# --- Main script ---
if __name__ == "__main__":
    try:
        # 1. Read the customers.csv file
        data = pd.read_csv('Case_study/data/customers.csv')
        print("Successfully loaded customers.csv")

        # 2. Select the 'Adresse' column
        if 'Adresse' in data.columns:
            addresses = data['Adresse'].dropna().unique()

            # 3. Geocode addresses
            geocoded_locations = geocode_addresses(addresses)

            # 4. Create a DataFrame with the results
            geo_df = pd.DataFrame(geocoded_locations, columns=['Adresse', 'Latitude', 'Longitude'])

            print("\n--- Geocoding Results ---")
            print(geo_df)

            # 5. Append new geocoded addresses to geocoded_addresses.csv in the correct location
            output_filename = 'Case_study/data/geocoded_addresses.csv'
            try:
                existing_df = pd.read_csv(output_filename)
                # Only add addresses not already present
                new_df = geo_df[~geo_df['Adresse'].isin(existing_df['Adresse'])]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except FileNotFoundError:
                combined_df = geo_df

            # Optionally, add depot if not present
            if 'Depot' not in combined_df['Adresse'].values:
                depot_address = {
                    'Adresse': ['Depot'],
                    'Latitude': [62.4293036],
                    'Longitude': [6.3280543]
                }
                depot_df = pd.DataFrame(depot_address)
                combined_df = pd.concat([combined_df, depot_df], ignore_index=True)

            combined_df.to_csv(output_filename, index=False)
            print(f"\nSuccessfully updated geocoded addresses in '{output_filename}'")

        else:
            print("\nERROR: Column 'Adresse' not found in customers.csv!")

    except FileNotFoundError:
        print("ERROR: '../data/customers.csv' not found. Please make sure the file is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")