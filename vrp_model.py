import pandas as pd

def load_vrp_data(customers_file='customers.csv', vehicles_file='vehicles.csv'):
    
    customers_df = pd.read_csv(customers_file)
    vehicles_df = pd.read_csv(vehicles_file)
    # Set the vehicle 'Navn' (name) as the index for easy lookup
    vehicles_df.set_index('Navn', inplace=True)
    return customers_df, vehicles_df
    
def check_capacity_feasibility(route_customers_df, vehicle_name, vehicles_df):
    """
    Checks if the total demand of a route exceeds the capacity of the assigned vehicle.

    Args:
        route_customers_df (pd.DataFrame): A DataFrame containing the customer data for a specific route.
        vehicle_name (str): The name of the vehicle (e.g., 'small', 'medium').
        vehicles_df (pd.DataFrame): The DataFrame containing vehicle capacity information.

    Returns:
        bool: True if the route is feasible in terms of capacity, False otherwise.
    """
    if route_customers_df.empty:
        return True # An empty route is always feasible

    try:
        # Get the capacity constraints for the given vehicle name
        vehicle_capacities = vehicles_df.loc[vehicle_name]
        
        # Sum the demands for the customers on the route
        total_pallets = route_customers_df['PPL pr leveranse'].sum()
        total_volume = route_customers_df['Volum pr leveranse (m3)'].sum()
        # Note: Customer weight is in 'tonn', vehicle capacity is in 'KG'. We must convert.
        total_weight_kg = route_customers_df['Vekt pr leveranse (tonn)'].sum() * 1000
        
        print(f"\n--- Checking Feasibility for Vehicle '{vehicle_name}' ---")
        print(f"Route Demands: Pallets={total_pallets:.2f}, Volume={total_volume:.2f} m3, Weight={total_weight_kg:.2f} KG")
        print(f"Vehicle Limits: Pallets={vehicle_capacities['PPL total']:.2f}, Volume={vehicle_capacities['m3']:.2f} m3, Weight={vehicle_capacities['Vekt (KG)']:.2f} KG")

        # Check if any demand exceeds the vehicle's capacity
        if total_pallets > vehicle_capacities['PPL total']:
            print("Result: Infeasible - Pallet capacity exceeded.")
            return False
        if total_volume > vehicle_capacities['m3']:
            print("Result: Infeasible - Volume capacity exceeded.")
            return False
        if total_weight_kg > vehicle_capacities['Vekt (KG)']:
            print("Result: Infeasible - Weight capacity exceeded.")
            return False
            
        print("Result: Feasible")
        return True

    except KeyError:
        print(f"Error: Vehicle name '{vehicle_name}' not found in the vehicle data.")
        return False
    except Exception as e:
        print(f"An error occurred during feasibility check: {e}")
        return False

if __name__ == "__main__":
    customers_df, vehicles_df = load_vrp_data()
    
    if customers_df is not None and vehicles_df is not None:
        print("\n--- Vehicle Capacity Information ---")
        print(vehicles_df)

        # --- Example Demonstration ---
        # Create a simple example route with the first 5 customers
        example_route_df = customers_df.head(5)
        
        # Check feasibility for a 'small' vehicle
        check_capacity_feasibility(example_route_df, vehicle_name='small', vehicles_df=vehicles_df)
        
        # Check feasibility for a 'large' vehicle
        check_capacity_feasibility(example_route_df, vehicle_name='large', vehicles_df=vehicles_df)
        
        # --- Example of an obviously infeasible route ---
        # Try all customers on one route with the largest vehicle
        infeasible_route_df = customers_df
        check_capacity_feasibility(infeasible_route_df, vehicle_name='large', vehicles_df=vehicles_df)
