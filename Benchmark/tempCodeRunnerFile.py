 destroyed_sol = destroy_ops[d_idx](
            current_sol, 
            n_remove, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx
        )
        
        # Apply Repair (Pass neighbor_sets)
        repaired_sol = repair_ops[r_idx](
            destroyed_sol, 
            distance_matrix_array=dist_matrix, 
            customer_addr_idx=cust_addr_idx,
            customer_arrays=cust_arrays,
            vehicles_df=vehicles_df,
            neighbor_sets=neighbor_sets
        )