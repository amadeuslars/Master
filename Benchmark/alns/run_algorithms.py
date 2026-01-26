import subprocess
import time
import re
import os
import statistics

# Configuration
SCRIPTS = ["alns_SA.py", "alns_RRT.py"]
NUM_RUNS = 10
OUTPUT_FILE = "results.txt"

def run_algorithm_realtime(script_name, run_index):
    """
    Runs the python script, streams 'New Best' lines to console immediately,
    captures full output for parsing later.
    """
    start_time = time.time()
    
    # -u flag forces unbuffered output so we see prints immediately
    cmd = ["python", "-u", script_name]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge errors into stdout
        text=True,
        bufsize=1 # Line buffered
    )
    
    captured_lines = []
    
    # Read output line by line as it is generated
    for line in process.stdout:
        line_clean = line.strip()
        captured_lines.append(line_clean)
        
        # Check if this line indicates a new best solution
        # Pattern matches: "Iter 123 [New Best]: ..."
        if "[New Best]" in line_clean:
            print(f"    [Run {run_index}] >> {line_clean}")
            
    # Wait for process to close
    process.wait()
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Reassemble full output for parsing
    full_output = "\n".join(captured_lines)
    
    # --- Parsing Logic (Same as before) ---
    
    # 1. Get the Final Best Cost reported at the end
    final_cost_match = re.search(r"Best Cost:\s*(\d+\.\d+)", full_output)
    
    if final_cost_match:
        best_cost = float(final_cost_match.group(1))
    else:
        best_cost = float('inf') 

    # 2. Find which iteration produced this cost
    found_on_iter = "Warmup/Init"
    iter_pattern = re.compile(r"Iter (\d+) \[New Best\]:\s*(\d+\.\d+)")
    
    for line in captured_lines:
        match = iter_pattern.search(line)
        if match:
            it_num = match.group(1)
            cost_at_iter = float(match.group(2))
            
            # Match strictly against the final reported cost
            if abs(cost_at_iter - best_cost) < 0.001:
                found_on_iter = it_num
                
    return best_cost, found_on_iter, runtime

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.2f}s"

def main():
    results_buffer = []

    print(f"Starting Benchmark ({NUM_RUNS} runs per algorithm)...")
    print("Real-time updates will appear below for every new global best found.")
    print("-" * 60)

    for script in SCRIPTS:
        print(f"\nBenchmarking {script}...")
        print("=" * 60)
        
        script_results = []
        script_costs = []
        
        # Header for the text file
        script_results.append(f"{script.replace('.py', '')}:")
        
        for i in range(1, NUM_RUNS + 1):
            print(f"Starting Run {i}/{NUM_RUNS}...")
            
            try:
                # This function now prints to terminal while running
                best_cost, found_iter, runtime_sec = run_algorithm_realtime(script, i)
                
                script_costs.append(best_cost)
                
                # Create the result string
                line = (f"Run {i}: Best found solution: {best_cost:.2f} | "
                        f"Found on iteration: {found_iter} | "
                        f"Runtime: {format_time(runtime_sec)}")
                
                script_results.append(line)
                print(f"Run {i} Finished. Final Cost: {best_cost:.2f}")
                print("-" * 30)
                
            except Exception as e:
                print(f"Run {i} Failed! ({e})")
                script_results.append(f"Run {i}: Failed with error {e}")

        # Calculate Average
        if script_costs:
            avg_cost = statistics.mean(script_costs)
            script_results.append(f"Average best found solution: {avg_cost:.2f}")
        else:
            script_results.append("Average best found solution: N/A")
            
        script_results.append("") # Spacer
        results_buffer.extend(script_results)

    # Write final report
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results_buffer))
        
    print("\n" + "="*60)
    print(f"Benchmark complete. Results saved to {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()