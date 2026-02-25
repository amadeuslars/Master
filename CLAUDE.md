# Master Thesis: AI-Driven Operator Selection for VRPTW

## Project Overview
Comparing DRLH vs ALNS-RRT vs ALNS-URS on VRPTW. Two workstreams:
1. **Benchmark** — Complete. DRLH and ALNS-RRT competitive, both beat ALNS-URS baseline on Homberger-600 (1000 iter, 10 runs).
2. **Case_study** — In progress. Real-world routing for HI Giørtz (Ålesund), 40 customers, 15 vehicles.

## Repository Structure
```
├── Benchmark/           # Complete (ALNS-RRT, ALNS-URS, DRLH on Solomon/Homberger)
├── Case_study/
│   ├── alns/alns.py     # Main ALNS loop (RRT + adaptive roulette wheel)
│   ├── alns/visualize.py  # Folium map output with OSRM road geometry
│   ├── utils/           # operators, feasibility, cost, utils, read_data, preprocess
│   └── data/            # customers.csv, vehicles.csv, time_matrix.csv, distance_matrix.csv
└── logs/                # Benchmark result CSVs + trained PPO model checkpoints
```

## Case Study: HI Giørtz Delivery Routing
- **40 customers** + depot in Ålesund, Norway
- **5 vehicle types**: small (17.5 PPL) to large (33 PPL), 3 of each = 15 vehicles
- **Objective**: Minimize total travel time

### Constraints (implemented)
- Multi-dimensional capacity: PPL total, PPL freezer, volume (m3), weight (kg)
- Time windows per customer; max route duration 8h
- Lunch break (30 min) must fit feasibly in route
- Vehicle-store compatibility: Biltype 2 stores → small/medium only (PPL ≤ 22)

### ALNS Status
- Framework: RRT acceptance + adaptive roulette wheel, 5 destroy + 2 repair operators — all working
- Initial solution: all-unassigned (greedy constructor — Stage 1 in progress)
- Lunch break: earliest-valid position (noon-targeting — Stage 1 in progress)
- Output: console only (schedule export + map — Stage 2 planned)

### Key Files
- `Case_study/alns/alns.py` — ALNS loop
- `Case_study/alns/visualize.py` — Folium/OSRM map generation
- `Case_study/utils/operators.py` — 5 destroy + 2 repair operators
- `Case_study/utils/feasibility.py` — All constraint checks
- `Case_study/utils/cost.py` — Travel time calculation
- `Case_study/utils/utils.py` — Solution class, data loading, initial solution

## Key Conventions
- Customer indices: 1-based; depot: index 0
- `addr_idx` and `customer_arrays`: 0-based (use `c-1` for customer c)
- Time matrix: hours (converted /3600 on load); time windows in absolute hours from midnight
- Cost = travel time (case study) or travel distance (benchmark)
- `vehicles_dict`: vehicle_name → `{PPL total, PPL Frys, m3, Vekt (KG)}`
- Feasibility check order: capacity → time windows → vehicle compatibility

## TODO
- **Service/loading time estimates**: Static defaults (service=20min, load=1h, deload=20min). Per-customer and per-vehicle estimates to be sourced from company data later.
- **Overcapacity/trailers**: Deferred. If unassigned customers remain after solve, flag them clearly in output. Trailer recommendation logic to be added in a later stage.
- **Shift structure**: Two shifts (early: lunch ~12:00, late: lunch ~18:00). Shift assignment per driver/route not yet in data. Currently defaulting to noon for all routes.

## Claude Code Usage
- Be sparse with token usage — avoid reading files unnecessarily
- Ask before doing broad codebase exploration
- Prefer targeted reads of specific files over scanning many files
