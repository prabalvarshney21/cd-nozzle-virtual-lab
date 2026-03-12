# Virtual Aerodynamics Lab — Convergent-Divergent Nozzle
## Complete Simulation Suite for Mechanical Engineering Students

---

## Overview

This is a **Python-based virtual laboratory** for studying compressible flow through a
Convergent-Divergent (De Laval) nozzle. All experiments are grounded in **exact 1-D
isentropic flow theory** and **Rankine-Hugoniot normal shock relations**.

Students modify clearly marked parameter blocks, run experiments, and record results
in their lab notebooks — exactly as they would in a physical wind tunnel facility.

---

## File Structure

```
nozzle_lab/
│
├── nozzle_physics.py                        ← Physics engine (DO NOT MODIFY)
│
├── experiment_01_mach_distribution.py       ← Exp 1: Mach color map
├── experiment_02_normal_shock.py            ← Exp 2: Shock location & entropy
├── experiment_03_throat_sonic_choking.py    ← Exp 3: Choking & mass flow
├── experiment_04_full_flow_field.py         ← Exp 4: Full P/T/ρ/V/M fields
├── experiment_05_nozzle_design_optimization.py ← Exp 5: Thrust & area ratio
│
├── run_all_experiments.py                   ← Run all or specific experiments
└── README.md                                ← This file
```

---

## Quick Start

```bash
# Run a single experiment
python experiment_01_mach_distribution.py

# Run all 5 experiments
python run_all_experiments.py

# Run experiments 2 and 4 only
python run_all_experiments.py --exp 2 4
```

Each script saves a `.png` figure and prints numerical results to the terminal.

---

## Experiment Summary

| # | Title | Key Concept | Student Changes |
|---|-------|-------------|-----------------|
| 1 | Mach Distribution & Color Map | A/A* relation, flow regimes | `AR_exit`, `Pb_cases` |
| 2 | Normal Shock Location | Rankine-Hugoniot, entropy | `P0`, `AR_exit`, `Pb_LIST` |
| 3 | Throat Sonic Choking | Mass flow saturation | `P0`, `T0`, `A_THROAT` |
| 4 | Full Flow Field | Isentropic coupling P/T/ρ/V | `P0`, `T0`, `AR_exit` |
| 5 | Design Optimisation | Thrust, Isp, altitude | `P0`, `T0`, altitudes |

---

## Physics Reference

### Isentropic Relations
```
T₀/T   = 1 + (γ-1)/2 · M²
P₀/P   = [1 + (γ-1)/2 · M²]^(γ/(γ-1))
ρ₀/ρ   = [1 + (γ-1)/2 · M²]^(1/(γ-1))
A/A*   = (1/M)·[(2/(γ+1))·(1 + (γ-1)/2·M²)]^((γ+1)/(2(γ-1)))
```

### Normal Shock (Rankine-Hugoniot)
```
M₂²     = [(γ-1)M₁² + 2] / [2γM₁² - (γ-1)]
P₂/P₁   = 1 + 2γ/(γ+1)·(M₁² - 1)
T₂/T₁   = (P₂/P₁)·(2 + (γ-1)M₁²) / ((γ+1)M₁²)
ρ₂/ρ₁   = (γ+1)M₁² / (2 + (γ-1)M₁²)
```

### Choked Mass Flow
```
ṁ_max = A*·P₀/√T₀ · √(γ/R) · [2/(γ+1)]^((γ+1)/(2(γ-1)))
```

### Thrust
```
F = ṁ·V_exit + (P_exit - P_atm)·A_exit
Cf = F / (P₀·A*)
Isp = F / (ṁ·g₀)
```

---

## Requirements

```
Python >= 3.8
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```

---

## Lab Notebook Tasks

Each experiment file contains a `STUDENT TASKS` block at the bottom with:
- Manual calculation tasks (verify against code output)
- Parameter variation studies  
- Discussion questions
- Graph-paper plotting assignments

Students should complete all tasks before the lab debrief session.

---

## Adjustable Parameters (per experiment)

All student-adjustable parameters are marked with:
```python
#  ██ STUDENT-ADJUSTABLE PARAMETERS  ██
```
**Only modify values inside this clearly labelled block.**

---

## Tips for Students

1. **Run Exp 1 first** — establishes the four flow regimes visually.
2. **Exp 2** is where most students are surprised by shock entropy loss.
3. **Exp 3** clearly shows *why* choking is independent of back pressure.
4. **Exp 4** ties all thermodynamic properties together simultaneously.
5. **Exp 5** connects theory to real rocket/jet engine design decisions.

---

*Physics engine based on: Anderson, J.D. "Modern Compressible Flow", 3rd Ed., McGraw-Hill.*
