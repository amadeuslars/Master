# Master Thesis: AI-Driven Operator Selection for VRPTW

## Project Overview
Comparing DRLH vs ALNS-RRT on VRPTW. Two workstreams:
1. **Benchmark** — Complete. DRLH and ALNS-RRT competitive, both beat ALNS-URS baseline on Homberger-600 (1000 iter, 10 runs).
2. **Case_study** — In progress. Real-world routing for HI Giørtz (Ålesund), 300+ customers across 5 delivery days, 18 vehicles. ALNS-RRT solver working with full multi-trip/shift/lunch constraints. DRLH zero-shot transfer solver created for comparison.

## Repository Structure
```
├── Benchmark/           # Complete (ALNS-RRT, ALNS-URS, DRLH on Solomon/Homberger)
├── Case_study/
│   ├── alns/
│   │   ├── alns.py            # Main ALNS loop (RRT + adaptive roulette wheel)
│   │   ├── solve_drlh.py      # DRLH zero-shot transfer (pre-trained PPO on case study data)
│   │   ├── hig_solution.py    # HIG current-solution loader (for comparison)
│   │   └── visualize.py       # Folium map output with OSRM road geometry
│   ├── utils/
│   │   ├── operators.py          # 3 destroy + 2 repair operators (Python)
│   │   ├── operators_cy.pyx      # Cython-compiled operators + feasibility (production)
│   │   ├── setup_cy.py           # Cython build script
│   │   ├── feasibility.py        # All constraint checks (Python fallback)
│   │   ├── cost.py               # Travel time calculation
│   │   ├── utils.py              # Solution class, data loading, initial solution
│   │   ├── preprocess_data.py    # Merge daily CSVs → unified customers.csv
│   │   ├── geocode_addresses.py  # Geocode missing addresses via Nominatim
│   │   └── generate_matrices.py  # Per-day OSRM matrix generation (chunked)
│   └── data/
│       ├── raw/                  # Drop daily delivery CSVs here
│       ├── customers.csv         # Unified customer file (all days, English columns)
│       ├── customers_alesund_sula_tue.csv  # Subset: Ålesund+Sula Tuesday (~108 customers)
│       ├── customers_old.csv     # Legacy 40-customer data (for cross-reference)
│       ├── vehicles.csv          # Vehicle fleet (5 types, 18 instances)
│       └── matrices/             # Per-day OSRM matrices
│           ├── mon/time_matrix.csv, distance_matrix.csv
│           ├── tue/ ...
│           ├── wed/ ...
│           ├── thu/ ...
│           └── fri/ ...
└── logs/                # Benchmark result CSVs + trained PPO model checkpoints
```

## Data Pipeline (run in order)
1. `python Case_study/utils/preprocess_data.py` — merge raw CSVs → customers.csv
2. `python Case_study/utils/geocode_addresses.py` — fill in missing lat/lon
3. `python Case_study/utils/generate_matrices.py [day...]` — OSRM matrices per day

## Case Study: HI Giørtz Delivery Routing
- **300+ customers** across 5 delivery days (~60/day with overlap) + depot in Ålesund
- **5 vehicle types**: small (17.5 PPL) to large (33 PPL), fleet of 15
- **Objective**: Minimize total travel time

### New Data Schema (customers.csv)
| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | int | Unique ID (from kundenr) |
| `customer_name` | str | Business name |
| `address` | str | Street address |
| `postal_code` | str | Postal code |
| `latitude` / `longitude` | float | WGS84 coordinates |
| `geocode_status` | str | `found` / `manual` / `missing` |
| `delivery_day` | str | Single day: `mon`, `tue`, etc. (one row per customer-day) |
| `tw_start` / `tw_end` | str | Time window `HH:MM` |
| `ppl` | float | PPL per delivery |
| `ppl_freeze` | float | PPL freeze |
| `volume_m3` | float | Volume in m3 |
| `weight_kg` | float | Weight in kg |
| `vehicle_type` | int | 1=any, 2=small/medium only |
| `tour_id` | str | Historic route ID |
| `service_time_min` | float | Service time in minutes (default 20) |

### Shift & Trip Structure
- **Two shifts per day**: early (~06:00–14:00), late (~14:00–22:00)
- **Max 2 trips per shift** → up to 4 trips per vehicle per day
- **15 vehicles** → up to 60 trips/day theoretical max
- **Timeline per trip**: load at depot (~1h) → deliver → return to depot → deload (~30 min)
- **Lunch break**: 30 min, must occur within first 6 hours of shift
- **Loading time**: ~1h at depot before departure (may vary by vehicle type later)
- **Deloading time**: ~30 min at depot when returning between trips
- **Service time**: ~20 min per customer stop (default)

### Constraints (all implemented)
- Multi-dimensional capacity: PPL total, PPL freezer, volume (m3), weight (kg) — with 1/3 PPL tolerance
- Time windows per customer (absolute hours from midnight)
- Vehicle-store compatibility: vehicle_type 2 → small/medium only (PPL ≤ 22)
- Multi-trip: 4 trips per vehicle per day (2 shifts × 2 trips), depot return with deload/reload
- Shift-aware scheduling: S1 (06:00–14:00), S2 (15:00–23:00), trip-2 departs after trip-1 returns
- Lunch break: 30 min, must occur within first 6h of shift start (trip-1 only)
- Loading time: 1h at depot before each trip departure
- Deloading time: 20 min at depot after return
- Two-phase repair: trip-1 routes filled first, trip-2 validated against updated trip-1 departure
- Safety net: `validate_and_trim_routes` trims overflowing routes post-repair

### ALNS Status
- Framework: RRT acceptance (threshold=10%) + adaptive roulette wheel — working
- 3 destroy operators: random, worst, cluster removal
- 2 repair operators: greedy insertion, regret-2 insertion (both two-phase)
- 5 fixed removal size buckets: xs(2-5), sm(5-10), md(10-20), lg(20-30), xl(30-40)
- Local search: 2-opt + or-opt available but currently commented out for fair DRLH comparison
- Cython backend: `operators_cy.pyx` compiled for ~10x speedup; Python fallback if not compiled
- `run_alns(delivery_day='tue', customers_file='...')` — configurable day and customer file
- Initial solution: all customers in dummy route, ALNS assigns them via repair operators

### DRLH Status
- Zero-shot transfer: pre-trained PPO agent (`5310_all_files_agent`) on case study data
- `CaseStudyDRLHEnv` wraps case study operators with 21D state / 30-action Gymnasium interface
- Same operators, buckets, RRT threshold as ALNS for apples-to-apples comparison
- Agent trained on Solomon/Homberger benchmark instances, tested on real-world out-of-distribution data

### Key Files
- `Case_study/alns/alns.py` — ALNS-RRT loop
- `Case_study/alns/solve_drlh.py` — DRLH zero-shot transfer solver
- `Case_study/alns/visualize.py` — Folium/OSRM map (reads lat/lon from customers_df)
- `Case_study/utils/utils.py` — `load_vrp_data(delivery_day)` with legacy fallback
- `Case_study/utils/operators_cy.pyx` — Cython operators + feasibility (production)
- `Case_study/utils/operators.py` — Python operators (fallback/reference)

## Key Conventions
- Customer indices: 1-based; depot: index 0
- `addr_idx` and `customer_arrays`: 0-based (use `c-1` for customer c)
- Time matrix: hours (converted /3600 on load); time windows in absolute hours from midnight
- Cost = travel time (case study) or travel distance (benchmark)
- `vehicles_dict`: vehicle_name → `{PPL total, PPL Frys, m3, Vekt (KG)}`
- Feasibility check order: capacity → time windows → vehicle compatibility
- `load_vrp_data()` auto-detects old vs new schema via column name check
- `customers.csv`: one row per customer per delivery day (per-day quantities preserved)
- Matrix labels: coordinate-based (`lat_lon` format) for robust lookup

## TODO
- **Greedy constructor**: Build feasible initial solution (assign customers to routes before ALNS starts). Currently all customers start in dummy. Nice-to-have but important for solution quality.
- **Matrix regeneration**: Some addresses updated in `customers_alesund_sula_tue.csv`. Need to rerun `python Case_study/utils/generate_matrices.py tue --file Case_study/data/customers_alesund_sula_tue.csv` with OSRM server running.
- **Full dataset run**: Test ALNS on all 5 delivery days (not just Tuesday subset).
- **HIG comparison**: Validate `hig_solution.py` against company's actual routes for quality comparison.
- **Service/loading time estimates**: Static defaults (service=20min, load=1h, deload=20min). Per-customer and per-vehicle estimates to be sourced from company data later.
- **Overcapacity/trailers**: Deferred. If unassigned customers remain after solve, flag them clearly in output.

## Claude Code Usage
- Be sparse with token usage — avoid reading files unnecessarily
- Ask before doing broad codebase exploration
- Prefer targeted reads of specific files over scanning many files
