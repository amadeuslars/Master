"""
Read Solomon benchmark instances and generate CSV files similar to real-life data format.
Creates distance matrix and vehicle information as CSV files.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os


class SolomonInstance:
    """Parse and store Solomon benchmark instance data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.instance_name = None
        self.num_vehicles = None
        self.vehicle_capacity = None
        self.customers = []  # List of customer dictionaries
        self.depot = None
        
        self._parse_file()
    
    def _parse_file(self):
        """Parse Solomon instance file."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Get instance name
        self.instance_name = lines[0].strip()
        
        # Parse vehicle information
        for i, line in enumerate(lines):
            if 'VEHICLE' in line:
                capacity_line = lines[i + 2].strip().split()
                self.num_vehicles = int(capacity_line[0])
                self.vehicle_capacity = float(capacity_line[1])
                break
        
        # Parse customer data
        reading_customers = False
        
        for line in lines:
            if 'CUST NO.' in line:
                reading_customers = True
                continue
            
            if reading_customers and line.strip():
                parts = line.strip().split()
                if len(parts) >= 7:
                    customer = {
                        'id': int(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'demand': float(parts[3]),
                        'ready_time': float(parts[4]),
                        'due_date': float(parts[5]),
                        'service_time': float(parts[6])
                    }
                    
                    if customer['id'] == 0:
                        self.depot = customer
                    else:
                        self.customers.append(customer)
    
    def calculate_euclidean_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate Euclidean distance between two locations."""
        dx = loc1['x'] - loc2['x']
        dy = loc1['y'] - loc2['y']
        return np.sqrt(dx**2 + dy**2)
    
    def create_distance_matrix(self) -> pd.DataFrame:
        """
        Create distance matrix CSV similar to real-life data.
        Row 0/Column 0 is depot, then customers 1, 2, 3, ...
        """
        # All locations (depot + customers)
        all_locations = [self.depot] + self.customers
        n = len(all_locations)
        
        # Create distance matrix
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                distances[i, j] = self.calculate_euclidean_distance(
                    all_locations[i], 
                    all_locations[j]
                )
        
        # Create DataFrame with proper labels
        labels = ['Depot'] + [f'Customer_{c["id"]}' for c in self.customers]
        df = pd.DataFrame(distances, index=labels, columns=labels)
        
        return df
    
    def create_customers_csv(self) -> pd.DataFrame:
        """
        Create customers CSV similar to real-life data format.
        """
        customers_data = []
        
        for cust in self.customers:
            customers_data.append({
                'customer_id': cust['id'],
                'x_coord': cust['x'],
                'y_coord': cust['y'],
                'demand': cust['demand'],
                'ready_time': cust['ready_time'],
                'due_date': cust['due_date'],
                'service_time': cust['service_time']
            })
        
        return pd.DataFrame(customers_data)
    
    def create_vehicles_csv(self) -> pd.DataFrame:
        """
        Create vehicles CSV similar to real-life data format.
        """
        vehicle_data = {
            'vehicle_type': ['Standard'],
            'capacity': [self.vehicle_capacity],
            'num_vehicles': [self.num_vehicles]
        }
        
        return pd.DataFrame(vehicle_data)
    
    def create_depot_csv(self) -> pd.DataFrame:
        """Create depot information CSV."""
        depot_data = {
            'location': ['Depot'],
            'x_coord': [self.depot['x']],
            'y_coord': [self.depot['y']],
            'ready_time': [self.depot['ready_time']],
            'due_date': [self.depot['due_date']]
        }
        
        return pd.DataFrame(depot_data)
    
    def export_to_csv(self, output_dir: str = None):
        """
        Export all data to CSV files.
        
        Args:
            output_dir: Directory to save CSV files. If None, uses instance name as folder.
        """
        if output_dir is None:
            output_dir = self.instance_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export distance matrix
        distance_matrix = self.create_distance_matrix()
        distance_matrix.to_csv(os.path.join(output_dir, 'distance_matrix.csv'))
        print(f"✓ Created {output_dir}/distance_matrix.csv ({len(distance_matrix)}x{len(distance_matrix)})")
        
        # Export customers
        customers_df = self.create_customers_csv()
        customers_df.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)
        print(f"✓ Created {output_dir}/customers.csv ({len(customers_df)} customers)")
        
        # Export vehicles
        vehicles_df = self.create_vehicles_csv()
        vehicles_df.to_csv(os.path.join(output_dir, 'vehicles.csv'), index=False)
        print(f"✓ Created {output_dir}/vehicles.csv ({self.num_vehicles} vehicles)")
        
        # Export depot
        depot_df = self.create_depot_csv()
        depot_df.to_csv(os.path.join(output_dir, 'depot.csv'), index=False)
        print(f"✓ Created {output_dir}/depot.csv")
        
        return output_dir
    
    def __repr__(self):
        return (f"SolomonInstance({self.instance_name}, "
                f"customers={len(self.customers)}, "
                f"vehicles={self.num_vehicles}, "
                f"capacity={self.vehicle_capacity})")


def load_solomon_instance(file_path: str) -> SolomonInstance:
    """Load a Solomon benchmark instance from file."""
    return SolomonInstance(file_path)


def process_all_instances(data_dir: str = 'data', output_base_dir: str = 'processed'):
    """
    Process all Solomon instances in a directory and create CSV files for each.
    
    Args:
        data_dir: Directory containing .txt benchmark files
        output_base_dir: Base directory for output CSV files
    """
    import glob
    
    # Get all .txt files
    txt_files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    print(f"Found {len(txt_files)} benchmark instances in {data_dir}/")
    print("=" * 60)
    
    for txt_file in sorted(txt_files):
        print(f"\nProcessing {os.path.basename(txt_file)}...")
        
        instance = load_solomon_instance(txt_file)
        instance_output_dir = os.path.join(output_base_dir, instance.instance_name)
        instance.export_to_csv(instance_output_dir)
    
    print("\n" + "=" * 60)
    print(f"All instances processed! CSV files saved in {output_base_dir}/")


if __name__ == "__main__":
    # Example: Load and process a single instance
    print("=" * 60)
    print("Example: Loading C101 instance")
    print("=" * 60)
    
    instance = load_solomon_instance('data/c101.txt')
    print(f"\n{instance}")
    print(f"\nDepot: x={instance.depot['x']}, y={instance.depot['y']}")
    print(f"First 5 customers:")
    for c in instance.customers[:5]:
        print(f"  Customer {c['id']}: demand={c['demand']}, "
              f"window=[{c['ready_time']}, {c['due_date']}], "
              f"coords=({c['x']}, {c['y']})")
    
    # Create CSV files
    print("\n" + "=" * 60)
    print("Creating CSV files for C101...")
    print("=" * 60)
    instance.export_to_csv()
    
    # Show sample of distance matrix
    print("\n" + "=" * 60)
    print("Sample Distance Matrix (first 5x5):")
    print("=" * 60)
    dist_matrix = instance.create_distance_matrix()
    print(dist_matrix.iloc[:5, :5].round(2))
    
    # Show vehicles info
    print("\n" + "=" * 60)
    print("Vehicles Information:")
    print("=" * 60)
    print(instance.create_vehicles_csv())

