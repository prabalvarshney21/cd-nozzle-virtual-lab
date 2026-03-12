"""
experiment_05_nozzle_design_optimization.py
=============================================
EXPERIMENT 5 — Nozzle Design Optimisation & Thrust Performance
===============================================================

OBJECTIVE:
    Given fixed stagnation conditions (P₀, T₀) and ambient pressure P_atm,
    find the OPTIMUM area ratio AR_exit that maximises thrust coefficient Cf.
    Analyse under-expanded vs over-expanded penalties.

THEORY:
    Thrust equation:
        F = ṁ·V_exit + (P_exit - P_atm)·A_exit

    Thrust coefficient:
        Cf = F / (P₀ · A*)

    Three operating modes:
        • Over-expanded  : P_exit < P_atm  → oblique shock at exit (thrust loss)
        • Perfectly exp. : P_exit = P_atm  → maximum thrust for given AR
        • Under-expanded : P_exit > P_atm  → expansion fans at exit (some loss)

    Optimum AR for altitude h:
        Solve P_exit(AR) = P_atm(h)  →  inverse isentropic P relation

STUDENT TASKS (see bottom of file):
    ① Find optimum AR for sea level, 10 km, 30 km, space
    ② Plot Cf vs AR for each altitude
    ③ Understand area ratio selection for rocket engines vs jet engines

RUN:
    python experiment_05_nozzle_design_optimization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nozzle_physics import (NozzleFlowSolver, mach_from_area_ratio,
                             isentropic_P_ratio, isentropic_T_ratio,
                             normal_shock_P0_ratio,
                             R_AIR, GAMMA_AIR, sonic_velocity)

# ══════════════════════════════════════════════════════════════════════════════
#  ██ STUDENT-ADJUSTABLE PARAMETERS  ██
# ══════════════════════════════════════════════════════════════════════════════

P0       = 1_000_000   # Stagnation pressure [Pa]  (rocket-like: 10–20 bar)
T0       = 2800.0      # Stagnation temp [K]        (rocket combustion ~2500-3500K)
A_STAR   = 0.01        # Throat area [m²]

# Altitudes to analyze
ALTITUDES = {
    'Sea Level (0 km)' : 101_325,
    '10 km (cruise)'   : 26_500,
    '30 km (stratosph)': 1_197,
    'Space (vacuum)'   : 1.0,       # near-zero
}

AR_RANGE = np.linspace(1.05, 30, 500)

# ══════════════════════════════════════════════════════════════════════════════

GAMMA = GAMMA_AIR

def exit_pressure(AR, P0, gamma=GAMMA):
    """P_exit for isentropic supersonic flow at given AR."""
    M_e = mach_from_area_ratio(AR, supersonic=True, gamma=gamma)
    return P0 / isentropic_P_ratio(M_e, gamma)

def exit_velocity(AR, T0, gamma=GAMMA):
    """V_exit for isentropic supersonic flow."""
    M_e = mach_from_area_ratio(AR, supersonic=True, gamma=gamma)
    T_e = T0 / isentropic_T_ratio(M_e, gamma)
    return M_e * sonic_velocity(T_e, gamma)

def mass_flow(P0, T0, A_star, gamma=GAMMA):
    """Choked mass flow [kg/s]."""
    return (A_star * P0 / np.sqrt(T0) *
            np.sqrt(gamma / R_AIR) *
            (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1))))

def thrust(AR, P0, T0, P_atm, A_star, gamma=GAMMA):
    """Gross thrust [N] — includes pressure thrust term."""
    Pe  = exit_pressure(AR, P0, gamma)
    Ve  = exit_velocity(AR, T0, gamma)
    mdot= mass_flow(P0, T0, A_star, gamma)
    Ae  = A_star * AR
    return mdot * Ve + (Pe - P_atm) * Ae

def thrust_coefficient(AR, P0, T0, P_atm, A_star, gamma=GAMMA):
    """Cf = F / (P0 · A*)"""
    F = thrust(AR, P0, T0, P_atm, A_star, gamma)
    return F / (P0 * A_star)

def optimum_AR(P0, P_atm, gamma=GAMMA):
    """AR such that P_exit = P_atm (perfect expansion)."""
    if P_atm >= P0:
        return 1.0
    # M_exit from isentropic P relation
    Pb_ratio = P_atm / P0
    # (P0/P)^((g-1)/g) = 1 + (g-1)/2 * M^2
    M_e = np.sqrt(2 / (gamma - 1) * ((P0 / P_atm)**((gamma - 1) / gamma) - 1))
    from nozzle_physics import area_mach_ratio
    return area_mach_ratio(M_e, gamma)


def run_optimisation(P0=1_000_000, T0=2800.0, A_STAR=0.01, ALTITUDES=None, AR_MAX=30.0):
    import io, base64
    if ALTITUDES is None:
        ALTITUDES = {
            'Sea Level (0 km)' : 101_325,
            '10 km (cruise)'   : 26_500,
            '30 km (stratosph)': 1_197,
            'Space (vacuum)'   : 1.0,
        }
    AR_RANGE = np.linspace(1.05, AR_MAX, 500)
    print("\n" + "═"*65)
    print("  EXPERIMENT 5 — Nozzle Design Optimisation")
    print("═"*65)
    print(f"\n  P₀ = {P0/1e6:.2f} MPa  |  T₀ = {T0:.0f} K  |  A* = {A_STAR:.4f} m²")
    print(f"  ṁ_choked = {mass_flow(P0, T0, A_STAR)*1000:.4f} g/s\n")

    results = {}
    print(f"  {'Altitude':<22} {'P_atm(Pa)':>10} {'AR_opt':>8} {'M_exit_opt':>11} {'Cf_max':>9}")
    print(f"  {'-'*65}")
    for alt_name, P_atm in ALTITUDES.items():
        AR_opt = optimum_AR(P0, P_atm)
        M_opt  = mach_from_area_ratio(AR_opt, supersonic=True) if AR_opt > 1.0 else 1.0
        Cf_opt = thrust_coefficient(AR_opt, P0, T0, P_atm, A_STAR)
        print(f"  {alt_name:<22} {P_atm:>10.1f} {AR_opt:>8.2f} {M_opt:>11.4f} {Cf_opt:>9.4f}")
        results[alt_name] = dict(P_atm=P_atm, AR_opt=AR_opt, M_opt=M_opt, Cf_max=Cf_opt)

    # ── Figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), facecolor='#0D1117')
    fig.suptitle(f"EXPERIMENT 5 — Nozzle Design Optimisation\n"
                 f"P₀={P0/1e6:.1f} MPa  |  T₀={T0:.0f} K  |  A*={A_STAR} m²",
                 fontsize=15, color='white', fontweight='bold', y=0.99)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                          left=0.06, right=0.97, top=0.94, bottom=0.05)

    axes_all = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes_all:
        ax.set_facecolor('#161B22')
        for sp in ax.spines.values(): sp.set_color('#30363D')
        ax.tick_params(colors='#8B949E', labelsize=9)
        ax.xaxis.label.set_color('#C9D1D9')
        ax.yaxis.label.set_color('#C9D1D9')

    ax_Cf, ax_Pe, ax_Ve, ax_thrst, ax_Isp, ax_tbl = axes_all

    cmap_alt  = plt.cm.viridis
    alt_list  = list(ALTITUDES.items())
    n_alt     = len(alt_list)
    col_list  = [cmap_alt(i / (n_alt - 1)) for i in range(n_alt)]

    # ── Cf vs AR ──────────────────────────────────────────────────────────
    for i, (alt_name, P_atm) in enumerate(alt_list):
        Cf_arr = np.array([thrust_coefficient(ar, P0, T0, P_atm, A_STAR)
                           for ar in AR_RANGE])
        col = col_list[i]
        ax_Cf.plot(AR_RANGE, Cf_arr, color=col, lw=2, label=alt_name)
        AR_opt = results[alt_name]['AR_opt']
        Cf_opt = results[alt_name]['Cf_max']
        ax_Cf.axvline(AR_opt, color=col, lw=1, ls=':', alpha=0.5)
        ax_Cf.scatter([AR_opt], [Cf_opt], color=col, s=60, zorder=5)

    ax_Cf.set_xlabel('Exit Area Ratio  A_e/A*', fontsize=10)
    ax_Cf.set_ylabel('Thrust Coefficient  Cf', fontsize=10)
    ax_Cf.set_title('Cf vs Area Ratio for Different Altitudes\n(★ = optimum AR)',
                    color='#58A6FF', fontsize=10, pad=6)
    ax_Cf.legend(fontsize=8, facecolor='#1C2128', labelcolor='white',
                 edgecolor='#30363D', loc='lower right')
    ax_Cf.set_xlim(AR_RANGE[0], AR_RANGE[-1])
    ax_Cf.grid(color='#21262D', lw=0.5)

    # ── Exit pressure vs AR ───────────────────────────────────────────────
    Pe_arr = np.array([exit_pressure(ar, P0) for ar in AR_RANGE])
    ax_Pe.semilogy(AR_RANGE, Pe_arr, color='#2196F3', lw=2, label='P_exit')
    for i, (alt_name, P_atm) in enumerate(alt_list):
        ax_Pe.axhline(P_atm, color=col_list[i], lw=1.2, ls='--',
                       label=f'{alt_name} ({P_atm:.0f} Pa)')
        AR_opt = results[alt_name]['AR_opt']
        if AR_opt < AR_RANGE[-1]:
            ax_Pe.axvline(AR_opt, color=col_list[i], lw=0.8, ls=':', alpha=0.5)

    ax_Pe.set_xlabel('A_e/A*', fontsize=10)
    ax_Pe.set_ylabel('Exit Pressure  P_exit  [Pa]', fontsize=10)
    ax_Pe.set_title('Exit Pressure vs Area Ratio\n(Intersection = perfect expansion)',
                    color='#58A6FF', fontsize=10, pad=6)
    ax_Pe.legend(fontsize=7.5, facecolor='#1C2128', labelcolor='white',
                 edgecolor='#30363D')
    ax_Pe.set_xlim(AR_RANGE[0], AR_RANGE[-1])
    ax_Pe.grid(color='#21262D', lw=0.5, which='both')

    # Shade over/under-expanded regions for sea level
    P_sl = ALTITUDES['Sea Level (0 km)']
    ax_Pe.fill_between(AR_RANGE, Pe_arr, P_sl, where=(Pe_arr < P_sl),
                        color='#F44336', alpha=0.1, label='Over-expanded')
    ax_Pe.fill_between(AR_RANGE, Pe_arr, P_sl, where=(Pe_arr > P_sl),
                        color='#4CAF50', alpha=0.1, label='Under-expanded')

    # ── Exit velocity vs AR ───────────────────────────────────────────────
    Ve_arr = np.array([exit_velocity(ar, T0) for ar in AR_RANGE])
    ax_Ve.plot(AR_RANGE, Ve_arr, color='#FF9800', lw=2)
    ax_Ve.set_xlabel('A_e/A*', fontsize=10)
    ax_Ve.set_ylabel('Exit Velocity  V_exit  [m/s]', fontsize=10)
    ax_Ve.set_title('Exit Velocity vs Area Ratio\n(Asymptotes to max for vacuum)',
                    color='#58A6FF', fontsize=10, pad=6)
    ax_Ve.set_xlim(AR_RANGE[0], AR_RANGE[-1])
    ax_Ve.grid(color='#21262D', lw=0.5)

    # Theoretical max V (all enthalpy to kinetic)
    V_max = np.sqrt(2 * GAMMA / (GAMMA - 1) * R_AIR * T0)
    ax_Ve.axhline(V_max, color='#FF6B6B', lw=1.5, ls='--',
                   label=f'V_max = √(2·cp·T₀) = {V_max:.0f} m/s')
    ax_Ve.legend(fontsize=9, facecolor='#1C2128', labelcolor='white', edgecolor='#30363D')

    # ── Thrust vs AR for each altitude ───────────────────────────────────
    for i, (alt_name, P_atm) in enumerate(alt_list):
        F_arr = np.array([thrust(ar, P0, T0, P_atm, A_STAR)
                          for ar in AR_RANGE])
        ax_thrst.plot(AR_RANGE, F_arr, color=col_list[i], lw=2, label=alt_name)
        AR_opt = results[alt_name]['AR_opt']
        F_opt  = thrust(AR_opt, P0, T0, P_atm, A_STAR)
        ax_thrst.scatter([AR_opt], [F_opt], color=col_list[i], s=60, zorder=5)

    ax_thrst.set_xlabel('A_e/A*', fontsize=10)
    ax_thrst.set_ylabel('Thrust  F  [N]', fontsize=10)
    ax_thrst.set_title('Thrust Force vs Area Ratio\n(Shows over-expansion thrust loss at sea level)',
                        color='#58A6FF', fontsize=10, pad=6)
    ax_thrst.legend(fontsize=8, facecolor='#1C2128', labelcolor='white',
                    edgecolor='#30363D')
    ax_thrst.set_xlim(AR_RANGE[0], AR_RANGE[-1])
    ax_thrst.grid(color='#21262D', lw=0.5)

    # ── Specific Impulse Isp ─────────────────────────────────────────────
    g0 = 9.81
    mdot_val = mass_flow(P0, T0, A_STAR)
    for i, (alt_name, P_atm) in enumerate(alt_list):
        F_arr  = np.array([thrust(ar, P0, T0, P_atm, A_STAR) for ar in AR_RANGE])
        Isp_arr = F_arr / (mdot_val * g0)
        ax_Isp.plot(AR_RANGE, Isp_arr, color=col_list[i], lw=2, label=alt_name)

    ax_Isp.set_xlabel('A_e/A*', fontsize=10)
    ax_Isp.set_ylabel('Specific Impulse  Isp  [s]', fontsize=10)
    ax_Isp.set_title('Specific Impulse vs Area Ratio\n(Isp = F / (ṁ·g₀)  — efficiency metric)',
                      color='#58A6FF', fontsize=10, pad=6)
    ax_Isp.legend(fontsize=8, facecolor='#1C2128', labelcolor='white',
                   edgecolor='#30363D')
    ax_Isp.set_xlim(AR_RANGE[0], AR_RANGE[-1])
    ax_Isp.grid(color='#21262D', lw=0.5)

    # ── Summary table ─────────────────────────────────────────────────────
    ax_tbl.axis('off')
    tbl_rows = []
    for alt_name, res in results.items():
        F_opt = thrust(res['AR_opt'], P0, T0, res['P_atm'], A_STAR)
        Isp   = F_opt / (mdot_val * 9.81)
        tbl_rows.append([
            alt_name[:22],
            f"{res['AR_opt']:.2f}",
            f"{res['M_opt']:.3f}",
            f"{res['Cf_max']:.4f}",
            f"{F_opt:.1f}",
            f"{Isp:.1f}"
        ])
    cols = ['Altitude','AR_opt','M_exit','Cf_max','F_opt (N)','Isp (s)']
    tbl = ax_tbl.table(cellText=tbl_rows, colLabels=cols,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1C2128' if r > 0 else '#21262D')
        cell.set_text_props(color='#C9D1D9' if r > 0 else '#58A6FF',
                            fontfamily='monospace')
        cell.set_edgecolor('#30363D')
    ax_tbl.set_title('Optimum Design Summary  →  Copy to Lab Notebook',
                      color='#58A6FF', fontsize=10, pad=8)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='#0D1117')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return [img_b64]


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENT TASKS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌─────────────────────────────────────────────────────────────────────────┐
│  LAB NOTEBOOK — Experiment 5                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  TASK 1: Fill in the optimum AR for each altitude from the table.       │
│    Plot AR_opt vs P_atm on a log scale. Describe the trend.            │
│                                                                         │
│  TASK 2: Why does a rocket engine designed for vacuum                   │
│    (e.g., AR=100) perform POORLY at sea level?                         │
│    Quantify the Cf loss compared to the optimum.                       │
│                                                                         │
│  TASK 3: Real engines:                                                  │
│    • Space Shuttle Main Engine: AR = 77.5 (vacuum)                     │
│    • Merlin 1D (SpaceX, sea level): AR = 16                            │
│    • Merlin Vacuum: AR = 165                                            │
│    Using this code, check what Cf each would achieve at sea level.     │
│                                                                         │
│  TASK 4: Change T0 from 1000 K to 4000 K.                              │
│    Does the optimum AR change? Does Isp change? Explain.               │
│                                                                         │
│  TASK 5: Define and compute thrust coefficient numerically:             │
│    Cf = F / (P0 · A*)   — verify your table values match.             │
└─────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == '__main__':
    run_optimisation()
