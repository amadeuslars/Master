# Gemini Codelens Report: Master Thesis on VRP

## Project Overview

This is a Master Thesis project that aims to solve a **Vehicle Routing Problem with Time Windows (VRPTW)**. The work is divided into two main components: a **Python implementation** of a novel optimization algorithm and a **LaTeX-based research paper**.

The core of the project is to develop and implement an **Adaptive Large Neighborhood Search (ALNS)** metaheuristic. The novel contribution of this thesis is the integration of an **AI component** to dynamically learn and optimize the weights for the ALNS "destroy" and "repair" operators.

The project uses real-world logistics data from a company named **HI Giørtz**, which includes store locations, delivery demands, and time windows from a central warehouse in Ålesund, Norway.

**Key Technologies:**
- **Python:** For the optimization algorithm.
  - `pandas`: For data loading and manipulation from the `data.xlsx` file.
- **LaTeX:** For the academic thesis document.

---

## Building and Running the Project

### 1. Python Optimization Algorithm

The Python code is responsible for data processing, and eventually, for implementing and running the ALNS algorithm.

**Setup:**

It is recommended to use a virtual environment. To set up the dependencies, run:
```bash
pip install -r requirements.txt
```

**Running the Data Loader:**

The entry point for data processing is `read_data.py`. To execute it:
```bash
python read_data.py
```
This script currently reads the `data.xlsx` file and prints the first few rows.

### 2. LaTeX Thesis Document

The thesis paper itself is written in LaTeX. The main source file is `master_thesis/problemstilling.tex`.

**Compiling the PDF:**

To compile the LaTeX document into a PDF, you will need a LaTeX distribution (like MiKTeX, TeX Live, or MacTeX). A common way to compile is using `pdflatex`. You may need to run it multiple times to resolve references.

```bash
# Navigate to the thesis directory
cd master_thesis

# Compile the document (run twice for cross-referencing)
pdflatex problemstilling.tex
pdflatex problemstilling.tex

# Or, using latexmk (recommended if available)
latexmk -pdf problemstilling.tex
```
*(Note: These commands are placeholders. Your specific LaTeX setup might require a different command.)*

---

## Development Conventions

- **Dependency Management:** Python dependencies are managed in the `requirements.txt` file. Any new packages should be added there.
- **Separation of Concerns:** The project maintains a clear separation between the implementation code (in the root directory) and the research paper (in the `master_thesis/` directory).
- **Data:** The raw data is stored in `data.xlsx` and is read by the Python scripts. The `.gitignore` file should probably be updated to exclude this file if it contains sensitive company information.
- **Roadmap:** The `README.md` file contains a high-level roadmap for the project, which should be updated as tasks are completed.
