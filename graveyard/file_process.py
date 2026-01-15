import pandas as pd
import numpy as np
import re

def process_file(input_filename):
    # Read the content of the text file
    with open(input_filename, 'r') as f:
        content = f.read()

    # --- 1. Extract Vehicle Information ---
    # Looks for: "VEHICLE" followed by "NUMBER ... CAPACITY ... 25 ... 200"
    # We use regex to find the two numbers under NUMBER and CAPACITY
    vehicle_match = re.search(r'VEHICLE[\s\S]+?(\d+)\s+(\d+)', content)
    if vehicle_match:
        num_vehicles = int(vehicle_match.group(1))
        capacity = float(vehicle_match.group(2))
    else:
        # Fallback defaults if parsing fails
        num_vehicles = 25
        capacity = 700.0

    df_vehicles = pd.DataFrame({
        'vehicle_type': ['Standard'],
        'capacity': [capacity],
        'num_vehicles': [num_vehicles]
    })
    df_vehicles.to_csv('Benchmark/C201/vehicles.csv', index=False)
    print("Generated vehicles.csv")

    # --- 2. Extract Customer and Depot Information ---
    lines = content.split('\n')
    data_rows = []
    parsing_customers = False
    
    for line in lines:
        # Start parsing after the header line
        if 'CUST NO.' in line:
            parsing_customers = True
            continue
        
        if parsing_customers and line.strip():
            # Split line by whitespace
            parts = line.split()
            # Ensure we have enough columns (CUST NO, X, Y, DEMAND, READY, DUE, SERVICE)
            if len(parts) >= 7:
                data_rows.append([float(x) for x in parts[:7]])

    # Create a main DataFrame for all nodes
    cols = ['id', 'x', 'y', 'demand', 'ready', 'due', 'service']
    df_nodes = pd.DataFrame(data_rows, columns=cols)

    # --- 3. Create depot.csv ---
    # Depot is usually the first node (id 0)
    depot_node = df_nodes.iloc[0]
    df_depot = pd.DataFrame({
        'location': ['Depot'],
        'x_coord': [depot_node['x']],
        'y_coord': [depot_node['y']],
        'ready_time': [depot_node['ready']],
        'due_date': [depot_node['due']]
    })
    df_depot.to_csv('Benchmark/C201/depot.csv', index=False)
    print("Generated depot.csv")

    # --- 4. Create customers.csv ---
    # Customers are all nodes except the first one
    df_customers = df_nodes.iloc[1:].copy()
    df_customers_out = pd.DataFrame({
        'customer_id': df_customers['id'].astype(int),
        'x_coord': df_customers['x'],
        'y_coord': df_customers['y'],
        'demand': df_customers['demand'],
        'ready_time': df_customers['ready'],
        'due_date': df_customers['due'],
        'service_time': df_customers['service']
    })
    df_customers_out.to_csv('Benchmark/C201/customers.csv', index=False)
    print("Generated customers.csv")

    # --- 5. Create distance_matrix.csv ---
    # Calculate Euclidean distance between all pairs (Depot + Customers)
    coords = df_nodes[['x', 'y']].values
    num_nodes = len(coords)
    dist_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            # Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    # Create headers: Depot, Customer_1, Customer_2, ...
    headers = ['Depot'] + [f'Customer_{int(i)}' for i in df_customers['id']]
    
    df_dist = pd.DataFrame(dist_matrix, columns=headers, index=headers)
    df_dist.to_csv('Benchmark/C201/distance_matrix.csv')
    print("Generated distance_matrix.csv")

# Run the function
if __name__ == "__main__":
    process_file('Benchmark/C201/c201.txt')