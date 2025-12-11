# Master Thesis: AI-Enhanced ALNS for the Vehicle Routing Problem

This repository contains the research and implementation for a Master Thesis focused on solving the Vehicle Routing Problem (VRP) using a hybrid metaheuristic approach.

## Core Objective

The primary objective of this thesis is to develop and evaluate a hybrid optimization approach for the Vehicle Routing Problem (VRP). We will implement an **Adaptive Large Neighborhood Search (ALNS)** framework and integrate an **AI-based mechanism** to dynamically learn and optimize the selection weights of its destroy and repair operators.

The model will be developed and tested using real-world data provided by a partner company.

## Methodology

1.  **ALNS Framework**: The core of the solution will be a powerful ALNS metaheuristic, which is well-suited for complex combinatorial optimization problems like the VRP.
2.  **AI-driven Operator Weight Optimization**: A machine learning component will be integrated to analyze the performance of different operators during the search, allowing the algorithm to adapt its strategy and converge on better solutions more efficiently.
3.  **Real-World Data**: The algorithm will be designed to handle practical constraints found in real-world logistics, based on the provided dataset including delivery time windows, quantities, and store locations.

## Project Structure

```
.
├── data.xlsx               # Raw data from the partner company (Not committed to Git)
├── read_data.py            # Script for loading and initial processing of the data
├── requirements.txt        # Python package dependencies
├── master_thesis/          # LaTeX source for the final thesis document
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Pip for package management

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  It is highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required packages. A `requirements.txt` file should be created to manage dependencies.
    ```bash
    pip install pandas
    # Add other libraries like numpy, scikit-learn, etc. as needed
    ```
    *(To generate `requirements.txt`, you can run `pip freeze > requirements.txt` after installing your packages.)*

## Usage

The first step is to read and process the data. This can be done by running the `read_data.py` script:

```bash
python read_data.py
```

## Project Roadmap

1.  [x] **Project Initialization**: Set up repository and initial files.
2.  [ ] **Data Ingestion & Preprocessing**: Fully load, clean, and structure the data from `data.xlsx`.
3.  [ ] **Geocoding & Distance Matrix**:
    -   Convert store addresses into geographic coordinates (latitude, longitude).
    -   Generate a distance/travel time matrix between all locations.
4.  [ ] **Solution Evaluation Module**:
    -   Develop functions to check the feasibility of a given solution (route).
    -   Create the objective function to calculate the total cost (e.g., travel distance/time) of a solution.
5.  [ ] **ALNS Framework Implementation**:
    -   Implement the core ALNS search loop.
    -   Develop a set of "destroy" and "repair" operators.
6.  [ ] **AI Weight Optimization**:
    -   Design and integrate the machine learning model for adaptively tuning operator weights.
7.  [ ] **Experimentation & Analysis**:
    -   Run the full algorithm on the dataset.
    -   Analyze the results and the performance of the AI-enhanced approach.
8.  [ ] **Thesis Write-up**:
    -   Document the research, methodology, and findings in the `master_thesis/` directory.

---
