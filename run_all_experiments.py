"""
run_all_experiments.py
========================
Virtual Lab Runner — C-D Nozzle Aerodynamics Suite
Run all 5 experiments in sequence and generate all output figures.

Usage:
    python run_all_experiments.py            # run all
    python run_all_experiments.py --exp 1    # run only experiment 1
    python run_all_experiments.py --exp 1 3  # run experiments 1 and 3
"""

import sys, os, importlib, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPERIMENTS = {
    1: ("experiment_01_mach_distribution",     "Mach Number Distribution & Color Map"),
    2: ("experiment_02_normal_shock",           "Normal Shock Location & Pressure Jump"),
    3: ("experiment_03_throat_sonic_choking",   "Throat Sonic Condition & Mass Flow Choking"),
    4: ("experiment_04_full_flow_field",        "Full Flow Field (P, T, ρ, V, M)"),
    5: ("experiment_05_nozzle_design_optimization", "Nozzle Design Optimisation & Thrust"),
}

ENTRY_POINTS = {
    1: "plot_mach_distribution",
    2: "shock_sweep",
    3: "run_choking_experiment",
    4: "run_flow_field",
    5: "run_optimisation",
}

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║         VIRTUAL AERODYNAMICS LAB — C-D NOZZLE SIMULATION            ║
║         Convergent-Divergent Nozzle: 5-Experiment Suite             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Exp 1 │ Mach Number Distribution & Color Map                       ║
║  Exp 2 │ Normal Shock — Location, Jump, Entropy Loss                ║
║  Exp 3 │ Throat Sonic Condition & Choking                           ║
║  Exp 4 │ Full Flow Field (P, T, ρ, V, M) — 2-D Color Maps          ║
║  Exp 5 │ Nozzle Design Optimisation & Thrust / Isp                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

def run_experiment(exp_num):
    mod_name, title = EXPERIMENTS[exp_num]
    fn_name = ENTRY_POINTS[exp_num]
    print(f"\n{'━'*68}")
    print(f"  ▶  EXPERIMENT {exp_num} — {title}")
    print(f"{'━'*68}")
    t0 = time.time()
    mod = importlib.import_module(mod_name)
    fn  = getattr(mod, fn_name)
    fn()
    print(f"  ✓  Experiment {exp_num} completed in {time.time()-t0:.1f}s")

def main():
    print(BANNER)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='*', type=int, default=None,
                        help='Experiment numbers to run (1-5). Default: all.')
    args = parser.parse_args()

    to_run = args.exp if args.exp else list(EXPERIMENTS.keys())

    for n in to_run:
        if n not in EXPERIMENTS:
            print(f"  ⚠  Unknown experiment number: {n}  (valid: 1–5)")
            continue
        try:
            run_experiment(n)
        except Exception as e:
            print(f"  ✗  Experiment {n} FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'═'*68}")
    print("  All selected experiments complete.")
    print("  Output PNG files saved in current directory.")
    print(f"{'═'*68}\n")

if __name__ == '__main__':
    main()
