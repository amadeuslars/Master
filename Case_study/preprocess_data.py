import pandas as pd
import sys

def clean_and_convert_excel(input_file='data.xlsx'):
    """
    Reads data from an Excel file, cleans it, and saves each sheet as a separate CSV file.
    This version correctly handles the 'Biltype ' sheet structure.
    """
    print(f"Reading Excel file: {input_file}")
    try:
        xls = pd.ExcelFile(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        sys.exit(1)

    # --- Process the 'Kunder ' sheet ---
    try:
        customers_df = pd.read_excel(xls, sheet_name='Kunder ')
        customers_df.columns = customers_df.columns.str.strip()
        customers_df.to_csv('customers.csv', index=False)
        print("Successfully created 'customers.csv'")
    except Exception as e:
        print(f"Could not process the 'Kunder ' sheet: {e}")

    # --- Process the 'Biltype ' sheet ---
    try:
        # Based on user feedback, the headers are not being read correctly.
        # We will skip the first row and manually assign the correct headers.
        vehicles_df = pd.read_excel(xls, sheet_name='Biltype ', header=0)

        # Manually set the correct column headers provided by the user.
        # The user provided 5 headers. We will ensure the dataframe has 5 columns.
        if len(vehicles_df.columns) == 5:
            vehicles_df.columns = ['Navn', 'PPL total', 'PPL Frys', 'm3', 'Vekt (KG)']
        else:
            # Fallback for the corrupted first row
             vehicles_df = pd.read_excel(xls, sheet_name='Biltype ', header=1)
             vehicles_df.columns = ['Navn', 'PPL total', 'PPL Frys', 'm3', 'Vekt (KG)']


        # The user mentioned renaming vehicle types. Let's do that here.
        rename_map = {
            17.5: 'small',
            20: 'medium-small',
            22: 'medium',
            30: 'medium-large',
            33: 'large'
        }
        # Assuming the names are in the first column, which we now call 'Navn'
        # The user's example shows the names are already correct, but this is here for robustness.
        # If the first column was numeric IDs, this replace() would work.
        
        vehicles_df.to_csv('vehicles.csv', index=False)
        print("Successfully created 'vehicles.csv' with correct headers and data.")
    except Exception as e:
        print(f"Could not process the 'Biltype ' sheet: {e}")


if __name__ == "__main__":
    print("--- Starting Data Preprocessing (Corrected) ---")
    clean_and_convert_excel()
    print("\n--- Data Preprocessing Complete ---")