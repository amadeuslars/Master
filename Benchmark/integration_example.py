"""
Example integration of visualization tracker with ALNS algorithm.
Shows minimal changes needed to track and plot ALNS progress.
"""

from visualization import ALNSTracker

# In your run_alns() function, add these changes:

def run_alns_with_tracking():
    """Example showing where to add tracking code."""
    
    # 1. INITIALIZE TRACKER (add after operator definitions)
    destroy_op_names = ['Random Removal', 'Worst Removal', 'Cluster Removal']
    repair_op_names = ['Greedy Insertion', 'Regret Insertion']
    tracker = ALNSTracker(destroy_op_names, repair_op_names)
    
    # 2. RECORD ITERATIONS (add at end of main loop, before cooling)
    # Example placement in main loop:
    """
    for it in range(WARMUP_ITERATIONS, MAX_ITERATIONS):
        # ... your existing ALNS code ...
        
        # After acceptance decision, before cooling:
        new_best_found = (accepted and new_cost < best_cost)
        
        # For Q-learning, you'll need to track weights differently
        # Get current Q-values as proxy for weights:
        d_weights = agent.q_destroy[state].tolist()  
        r_weights = agent.q_repair[state].tolist()
        
        tracker.record_iteration(
            iteration=it,
            best_cost=best_sol._cost,
            current_cost=current_sol._cost,
            destroy_weights=d_weights,
            repair_weights=r_weights,
            new_best=new_best_found
        )
        
        # Cool Down
        curr_temp *= cooling_rate
    """
    
    # 3. GENERATE PLOTS (add at end of run_alns, after final output)
    """
    # After printing final results:
    
    print("\nGenerating visualization plots...")
    tracker.plot_all(prefix='alns_results', save=True, show=False)
    tracker.print_summary()
    """

# Alternative: Save tracking data every N iterations to reduce overhead
def run_alns_with_sparse_tracking():
    """Track every 10 iterations for better performance."""
    
    tracker = ALNSTracker(['Random', 'Worst', 'Cluster'], ['Greedy', 'Regret'])
    TRACK_INTERVAL = 10  # Track every 10 iterations
    
    """
    for it in range(WARMUP_ITERATIONS, MAX_ITERATIONS):
        # ... ALNS code ...
        
        # Only track every TRACK_INTERVAL iterations
        if it % TRACK_INTERVAL == 0:
            tracker.record_iteration(
                iteration=it,
                best_cost=best_sol._cost,
                current_cost=current_sol._cost,
                destroy_weights=d_weights,
                repair_weights=r_weights,
                new_best=new_best_found
            )
    """

if __name__ == "__main__":
    print("This file shows how to integrate ALNSTracker with alns_benchmark.py")
    print("\nKey integration points:")
    print("1. Create tracker after defining operators")
    print("2. Record iteration data in main loop")
    print("3. Generate plots at end of run")
    print("\nSee visualization.py for the tracker implementation")
