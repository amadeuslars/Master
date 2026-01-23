import subprocess
import time
import re
import os
import statistics

# Configuration
SCRIPTS = ["alns_SA.py", "alns_RRT.py"]
NUM_RUNS = 10
OUTPUT_FILE = "results.txt"

def run_algorithm(script_name):
    """
    Runs the python script using subprocess, captures output, 
    and returns (best_cost, found_iteration, runtime).
    """
    start_time = time.time()
    
    # Run the script and capture stdout
    # python <script_name>
    process = subprocess.run(
        ["python", script_name], 
        capture_output=True, 
        text=True
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    
    output = process.stdout
    
    # --- Parsing Logic ---
    # 1. Get the Final Best Cost reported at the end
    final_cost_match = re.search(r"Best Cost:\s*(\d+\.\d+)", output)
    
    if final_cost_match:
        best_cost = float(final_cost_match.group(1))
    else:
        best_cost = float('inf') # Error or failed run

    # 2. Find which iteration produced this cost
    # Look for lines like: "Iter 123 [New Best]: 1234.56"
    # We look for the LAST occurrence that matches our final best_cost
    found_on_iter = "Warmup/Init" # Default if found before main loop
    
    iter_pattern = re.compile(r"Iter (\d+) \[New Best\]:\s*(\d+\.\d+)")
    
    for line in output.splitlines():
        match = iter_pattern.search(line)
        if match:
            it_num = match.group(1)
            cost_at_iter = float(match.group(2))
            
            # If this line matches our final best cost, update the iteration found
            # (We use abs difference for float comparison safety)
            if abs(cost_at_iter - best_cost) < 0.001:
                found_on_iter = it_num
                
    return best_cost, found_on_iter, runtime

def format_time(seconds):
    """Formats time seconds into Xm Ys or just Xs"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.2f}s"

def main():
    results_buffer = []

    print(f"Starting Benchmark ({NUM_RUNS} runs per algorithm)...")
    print("-" * 50)

    for script in SCRIPTS:
        print(f"Benchmarking {script}...")
        script_results = []
        script_costs = []
        
        # Header for the text file block
        script_results.append(f"{script.replace('.py', '')}:")
        
        for i in range(1, NUM_RUNS + 1):
            print(f"  > Run {i}/{NUM_RUNS}...", end="", flush=True)
            
            try:
                best_cost, found_iter, runtime_sec = run_algorithm(script)
                script_costs.append(best_cost)
                
                # Format the line
                line = (f"Run {i}: Best found solution: {best_cost:.2f} | "
                        f"Found on iteration: {found_iter} | "
                        f"Runtime: {format_time(runtime_sec)}")
                
                script_results.append(line)
                print(f" Done. (Cost: {best_cost:.2f})")
                
            except Exception as e:
                print(f" Failed! ({e})")
                script_results.append(f"Run {i}: Failed with error {e}")

        # Calculate Average
        if script_costs:
            avg_cost = statistics.mean(script_costs)
            script_results.append(f"Average best found solution: {avg_cost:.2f}")
        else:
            script_results.append("Average best found solution: N/A")
            
        script_results.append("") # Empty line between algos
        results_buffer.extend(script_results)

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results_buffer))
        
    print("-" * 50)
    print(f"Benchmark complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Ensure we are in the directory of the script to find the other files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()