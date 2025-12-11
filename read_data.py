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
        # 1. Read the Excel file
        data = pd.read_excel('data.xlsx')
        print("Successfully loaded data.xlsx")

        # 2. Select the 'Adresse' column
        if 'Adresse' in data.columns:
            addresses = data['Adresse'].dropna().unique()
            
            # 3. Geocode a sample of the addresses (e.g., the first 5)
            addresses = addresses
            geocoded_locations = geocode_addresses(addresses)
            
            # 4. Create a DataFrame with the results
            geo_df = pd.DataFrame(geocoded_locations, columns=['Adresse', 'Latitude', 'Longitude'])
            
            print("\n--- Geocoding Results ---")
            print(geo_df)

            # 5. Save the DataFrame to a CSV file
            output_filename = 'geocoded_addresses.csv'
            geo_df.to_csv(output_filename, index=False)
            print(f"\nSuccessfully saved geocoded addresses to '{output_filename}'")

        else:
            print("\nERROR: Column 'Adresse' not found in data.xlsx!")

    except FileNotFoundError:
        print("ERROR: 'data.xlsx' not found. Please make sure the file is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")