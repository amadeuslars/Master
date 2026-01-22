"""
Visualization tools for ALNS algorithm results.
Tracks and plots solution evolution and operator performance.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


class ALNSTracker:
    """Tracks ALNS search progress for visualization."""
    
    def __init__(self, destroy_op_names: List[str], repair_op_names: List[str]):
        """
        Initialize tracker.
        
        Args:
            destroy_op_names: Names of destroy operators
            repair_op_names: Names of repair operators
        """
        self.destroy_op_names = destroy_op_names
        self.repair_op_names = repair_op_names
        
        # Solution evolution
        self.iterations = []
        self.best_costs = []
        self.current_costs = []
        self.best_found_iterations = []  # Iterations where new best was found
        self.best_found_costs = []
        
        # Operator weights evolution
        self.destroy_weights_history = []
        self.repair_weights_history = []
        
    def record_iteration(self, iteration: int, best_cost: float, current_cost: float, 
                        destroy_weights: List[float], repair_weights: List[float],
                        new_best: bool = False):
        """
        Record data from one iteration.
        
        Args:
            iteration: Current iteration number
            best_cost: Best solution cost so far
            current_cost: Current solution cost
            destroy_weights: Current destroy operator weights
            repair_weights: Current repair operator weights
            new_best: Whether this iteration found a new best solution
        """
        self.iterations.append(iteration)
        self.best_costs.append(best_cost)
        self.current_costs.append(current_cost)
        self.destroy_weights_history.append(destroy_weights.copy())
        self.repair_weights_history.append(repair_weights.copy())
        
        if new_best:
            self.best_found_iterations.append(iteration)
            self.best_found_costs.append(best_cost)
    
    def plot_solution_evolution(self, save_path: str = None, show: bool = True):
        """
        Plot solution cost evolution over iterations.
        Shows best cost, current cost, and marks when new best solutions are found.
        
        Args:
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot best and current costs
        ax.plot(self.iterations, self.best_costs, 'b-', linewidth=2, 
                label='Best Solution', alpha=0.8)
        ax.plot(self.iterations, self.current_costs, 'gray', linewidth=0.5, 
                label='Current Solution', alpha=0.5)
        
        # Mark iterations where new best was found
        if self.best_found_iterations:
            ax.scatter(self.best_found_iterations, self.best_found_costs, 
                      color='red', s=80, zorder=5, marker='*',
                      label='New Best Found', alpha=0.8)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Solution Cost', fontsize=12)
        ax.set_title('ALNS Solution Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add improvement stats
        if len(self.best_costs) > 0:
            initial_cost = self.best_costs[0]
            final_cost = self.best_costs[-1]
            improvement = ((initial_cost - final_cost) / initial_cost) * 100
            
            stats_text = f'Initial: {initial_cost:.2f}\n'
            stats_text += f'Final: {final_cost:.2f}\n'
            stats_text += f'Improvement: {improvement:.1f}%'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Solution evolution plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_operator_weights(self, save_path: str = None, show: bool = True):
        """
        Plot operator weight evolution over iterations as relative usage (0-1).
        Creates two subplots: one for destroy operators, one for repair operators.
        
        Args:
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convert to numpy arrays for easier plotting
        destroy_weights = np.array(self.destroy_weights_history)
        repair_weights = np.array(self.repair_weights_history)

        # Normalize to probabilities (relative usage) with small epsilon to avoid divide-by-zero
        destroy_sums = destroy_weights.sum(axis=1, keepdims=True) + 1e-10
        repair_sums = repair_weights.sum(axis=1, keepdims=True) + 1e-10
        destroy_norm = destroy_weights / destroy_sums
        repair_norm = repair_weights / repair_sums
        
        # Define colors for operators
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(self.destroy_op_names), 
                                                    len(self.repair_op_names))))
        
        # Plot destroy operator relative usage
        for i, name in enumerate(self.destroy_op_names):
            ax1.plot(self.iterations, destroy_norm[:, i], 
                    label=name, linewidth=2, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Relative Usage (0-1)', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.set_title('Destroy Operator Relative Usage Evolution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot repair operator relative usage
        for i, name in enumerate(self.repair_op_names):
            ax2.plot(self.iterations, repair_norm[:, i], 
                    label=name, linewidth=2, color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Relative Usage (0-1)', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.set_title('Repair Operator Relative Usage Evolution', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Operator weights plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_all(self, prefix: str = 'alns', save: bool = True, show: bool = True):
        """
        Generate all plots.
        
        Args:
            prefix: Prefix for saved filenames
            save: Whether to save plots
            show: Whether to display plots
        """
        solution_path = f'{prefix}_solution_evolution.png' if save else None
        weights_path = f'{prefix}_operator_weights.png' if save else None
        
        self.plot_solution_evolution(save_path=solution_path, show=show)
        self.plot_operator_weights(save_path=weights_path, show=show)
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of the search.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.best_costs:
            return {}
        
        initial_cost = self.best_costs[0]
        final_cost = self.best_costs[-1]
        improvement = ((initial_cost - final_cost) / initial_cost) * 100
        
        return {
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'improvement_percent': improvement,
            'total_iterations': len(self.iterations),
            'new_best_found': len(self.best_found_iterations),
            'last_improvement_iter': self.best_found_iterations[-1] if self.best_found_iterations else 0
        }
    
    def print_summary(self):
        """Print summary statistics."""
        stats = self.get_summary_stats()
        
        if not stats:
            print("No data recorded yet.")
            return
        
        print("\n" + "="*50)
        print("ALNS SEARCH SUMMARY")
        print("="*50)
        print(f"Initial Cost:        {stats['initial_cost']:.2f}")
        print(f"Final Cost:          {stats['final_cost']:.2f}")
        print(f"Improvement:         {stats['improvement_percent']:.2f}%")
        print(f"Total Iterations:    {stats['total_iterations']}")
        print(f"New Best Found:      {stats['new_best_found']} times")
        print(f"Last Improvement:    Iteration {stats['last_improvement_iter']}")
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Simulate ALNS data
    import random
    
    destroy_ops = ['Random Removal', 'Worst Removal', 'Cluster Removal']
    repair_ops = ['Greedy Insertion', 'Regret Insertion']
    
    tracker = ALNSTracker(destroy_ops, repair_ops)
    
    # Simulate 1000 iterations
    best_cost = 1000.0
    current_cost = 1000.0
    d_weights = [1.0, 1.0, 1.0]
    r_weights = [1.0, 1.0]
    
    for i in range(1000):
        # Simulate cost changes
        current_cost = best_cost + random.uniform(-50, 100)
        new_best = False
        
        if random.random() < 0.05:  # 5% chance of improvement
            best_cost *= 0.98
            new_best = True
        
        # Simulate weight changes
        d_weights = [w + random.uniform(-0.1, 0.1) for w in d_weights]
        r_weights = [w + random.uniform(-0.1, 0.1) for w in r_weights]
        
        tracker.record_iteration(i, best_cost, current_cost, d_weights, r_weights, new_best)
    
    # Generate plots
    tracker.plot_all(prefix='example', save=True, show=False)
    tracker.print_summary()
    
    print("\nExample plots generated!")
