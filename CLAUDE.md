# Master Thesis: AI-Driven Operator Selection for VRPTW

## Project Overview

Master's thesis comparing learning-based operator selection strategies for the Vehicle Routing Problem with Time Windows (VRPTW). Two workstreams:

1. **Benchmark** - Theoretical comparison of DRLH vs ALNS-RRT vs ALNS-URS on standard Solomon/Homberger instances
2. **Case_study** - Real-world application for HI Giørtz (Ålesund logistics company) with complex operational constraints

## Repository Structure

```
/
├── Benchmark/                    # Theoretical work (complete)
│   ├── alns/                     # ALNS variants (RRT, URS, SA)
│   ├── drl/                      # Deep RL (PPO via Stable-Baselines3)
│   ├── utils/                    # Shared: operators, cost, feasibility, visualization
│   ├── data/homberger_{100,200,400,600,800}/  # Solomon/Homberger instances
│   └── Instances/                # Train/Test split
├── Case_study/                   # Real-world case study (in progress)
│   ├── alns/alns.py              # Main ALNS loop
│   ├── utils/                    # operators, feasibility, cost, read_data, preprocess, utils
│   └── data/                     # Company data (customers, vehicles, matrices)
├── logs/                         # Results CSVs and trained models
│   ├── drlh_results.csv
│   ├── rrt_results.csv
│   ├── urs_results.csv
│   └── drl_sb3_*/                # PPO model checkpoints
└── master_thesis/                # LaTeX thesis document
```

## Benchmark (Theoretical Work)

### Three Approaches Compared

| Approach | Operator Selection | Acceptance | Key File |
|----------|-------------------|------------|----------|
| **DRLH** | PPO policy (19D state -> 50 actions) | RRT threshold | `Benchmark/drl/solve_sb3.py` |
| **ALNS-RRT** | Adaptive roulette wheel (alpha=0.8) | RRT threshold | `Benchmark/alns/alns_RRT.py` |
| **ALNS-URS** | Uniform random (baseline) | RRT threshold | `Benchmark/alns/alns_URS.py` |

### ALNS Core Components

**Destroy Operators** (5): random, worst, cluster, shaw, least_used_vehicle
**Repair Operators** (2): greedy insertion, regret insertion
**Implementation**: `Benchmark/utils/operators.py` (also Cython version: `operators2_raw.pyx`)

### RRT Acceptance Criterion
```
threshold = 0.10 * (remaining_iterations / max_iterations) * best_cost
accept if new_cost < best_cost + threshold
```
Permissive early (up to +10% worse), strict late.

### DRLH State Space (19 features)
Reduced distance, optimality gap, scaled costs, RRT threshold, stagnation counter, one-hot destroy op (5), bucket index (5 removal sizes), repair op (binary), remaining budget, feasibility pressure.

### Action Space: 50 discrete = 5 destroy ops x 5 removal buckets x 2 repair ops

### Data: Homberger VRPTW Instances
- Format: Solomon `.TXT` (customer ID, X, Y, demand, ready_time, due_date, service_time)
- Sizes: 100, 200, 400, 600, 800 customers
- Categories: C (clustered), R (random), RC (mixed)
- Test set: Homberger-600

### Results (1000 iterations, 10 runs, Homberger-600)
- DRLH and ALNS-RRT competitive; both significantly beat ALNS-URS baseline

## Case Study (Real-World Application)

### Problem: HI Giørtz Delivery Routing
- **40 customers** + 1 depot in Ålesund, Norway
- **5 vehicle types**: small (17.5 PPL) to large (33 PPL), 3 of each = 15 vehicles
- **Objective**: Minimize total travel time

### Constraints (Current + Planned)
- **Multi-dimensional capacity**: PPL total, PPL freezer, volume (m3), weight (kg)
- **Time windows**: Per-customer delivery windows with service times
- **Vehicle-store compatibility**: Biltype 2 stores only accept small/medium vehicles (PPL <= 22)
- **Planned**: Driver break rules (30min break per 6h), max workday (8-10h), etc.

### Data Pipeline
1. Company Excel -> CSV (`data/customers.csv`, `data/vehicles.csv`)
2. Geocoding via Nominatim -> `data/geocoded_addresses.csv`
3. OSRM routing (Docker, port 5001) -> `data/time_matrix.csv`, `data/distance_matrix.csv`

### Key Files
- `Case_study/alns/alns.py` - Main ALNS loop
- `Case_study/utils/operators.py` - Destroy/repair operators
- `Case_study/utils/feasibility.py` - Capacity, time windows, vehicle compatibility
- `Case_study/utils/cost.py` - Travel time calculation
- `Case_study/utils/utils.py` - Data loading, Solution class, initial solution
- `Case_study/utils/read_data.py` - Raw CSV reading
- `Case_study/utils/preprocess_data.py` - Data preprocessing

### Solution Representation
```python
class Solution:
    routes: List[List[int]]   # Per-vehicle routes (customer indices), last = dummy
    vehicles: List[str]       # Vehicle IDs
    _cost: float              # Total travel time
```
Dummy vehicle (last route) holds unassigned customers with +100 penalty each.

### Current Status
- Data pipeline: Complete
- ALNS framework: Restructured, needs debugging
- Operators: Only random destroy + greedy repair implemented
- Acceptance: Greedy only (no RRT/SA yet)
- Goal: Port benchmark operators and learning approaches to this problem

## Development Notes

### Running Benchmark
```bash
cd Benchmark
python alns/alns_RRT.py    # ALNS with adaptive weights
python alns/alns_URS.py    # ALNS baseline
python drl/solve_sb3.py    # DRLH inference (requires trained model)
python drl/train_sb3.py    # Train PPO agent
```

### Key Conventions
- Customer indices are 1-based; depot is index 0
- `addr_idx` and `customer_arrays` are 0-based (use `c-1` for customer c), same convention as benchmark
- Time matrix values in seconds, converted to hours (/3600) in case study
- Cost = total travel distance (benchmark) or travel time (case study)
- Feasibility check order: capacity -> time windows -> vehicle compatibility
- `vehicles_dict` is keyed by vehicle name -> capacity dict (e.g. `vehicles_dict['small']['PPL total']`)

### Claude Code Usage
- Be sparse with token usage — avoid reading files unnecessarily
- Ask before doing broad codebase exploration
- Prefer targeted reads of specific files over scanning many files
