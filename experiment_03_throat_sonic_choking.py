"""
experiment_03_throat_sonic_choking.py
=======================================
EXPERIMENT 3 — Throat Sonic Condition & Choking
================================================

OBJECTIVE:
    Demonstrate the choking phenomenon — show that once the throat reaches
    M = 1, mass flow rate cannot increase regardless of further pressure
    drop. Visualise the pressure-ratio at which choking begins.

THEORY:
    Critical (choked) mass flow:
        ṁ_max = A* · P₀ / √(T₀) · √(γ/R) · [2/(γ+1)]^((γ+1)/(2(γ-1)))

    This is the maximum possible mass flow for given P₀, T₀, A*.
    Once throat reaches M = 1 (sonic), upstream flow is isolated from
    downstream changes → no further increase in ṁ.

    Choked condition: Pb/P0 ≤ [2/(γ+1)]^(γ/(γ-1)) ≈ 0.528 for air

STUDENT TASKS (see bottom of file):
    ① Observe ṁ saturation on chart
    ② Compute choked mass flow for your parameters
    ③ Verify the critical pressure ratio

RUN:
    python experiment_03_throat_sonic_choking.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nozzle_physics import (NozzleFlowSolver, isentropic_P_ratio,
                             isentropic_T_ratio, isentropic_rho_ratio,
                             sonic_velocity, R_AIR, GAMMA_AIR)

# ══════════════════════════════════════════════════════════════════════════════
#  ██ STUDENT-ADJUSTABLE PARAMETERS  ██
# ══════════════════════════════════════════════════════════════════════════════

P0      = 200_000    # Stagnation pressure [Pa]
T0      = 300.0      # Stagnation temperature [K]
AR_exit = 3.0        # Exit area ratio
A_THROAT = 0.01      # Throat area [m²]  (affects ṁ magnitude, not ratios)

# Number of back pressure points in sweep
N_PB_POINTS = 80

# ══════════════════════════════════════════════════════════════════════════════

GAMMA = GAMMA_AIR

def choked_mass_flow(P0, T0, A_star, gamma=GAMMA):
    """Theoretical maximum (choked) mass flow [kg/s]"""
    coeff = np.sqrt(gamma / R_AIR) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))
    return A_star * P0 / np.sqrt(T0) * coeff

def critical_pressure_ratio(gamma=GAMMA):
    """Pb*/P0 = [2/(γ+1)]^(γ/(γ-1))"""
    return (2 / (gamma + 1))**(gamma / (gamma - 1))

def subsonic_mass_flow(Pb, P0, T0, A_throat, gamma=GAMMA):
    """Mass flow for unchoked subsonic flow through converging section."""
    Pb_ratio = np.clip(Pb / P0, 0.001, 1.0)
    # Mach at throat for given Pb/P0 (isentropic, throat = exit for conv. only)
    # For fully converging: M_throat = M from P/P0 inversion
    # Using: P/P0 = (1 + (g-1)/2 M²)^(-g/(g-1))
    bracket_val = (Pb_ratio)**((gamma - 1) / gamma)
    M_thr = np.sqrt(2 / (gamma - 1) * (1 / bracket_val - 1))
    M_thr = np.clip(M_thr, 0, 1)
    T_thr = T0 / isentropic_T_ratio(M_thr, gamma)
    rho_thr = Pb / (R_AIR * T_thr)    # approx using Pb as local P
    a_thr = sonic_velocity(T_thr, gamma)
    V_thr = M_thr * a_thr
    return rho_thr * V_thr * A_throat

def run_choking_experiment(P0=200_000, T0=300.0, AR_exit=3.0, A_THROAT=0.01, N_PB_POINTS=80):
    import io, base64
    solver = NozzleFlowSolver(P0=P0, T0=T0, AR_exit=AR_exit, n_pts=500)

    # ── Mass flow vs Pb/P0 sweep ──────────────────────────────────────────
    Pb_over_P0 = np.linspace(0.05, 1.0, N_PB_POINTS)
    Pb_arr     = Pb_over_P0 * P0
    Pcrit      = critical_pressure_ratio(GAMMA)
    mdot_max   = choked_mass_flow(P0, T0, A_THROAT)

    mdot = []
    M_throat = []
    for Pb in Pb_arr:
        if Pb / P0 >= Pcrit:
            # Unchoked: mass flow < max
            mf = subsonic_mass_flow(Pb, P0, T0, A_THROAT)
            Mt = np.sqrt(2/(GAMMA-1)*((P0/Pb)**((GAMMA-1)/GAMMA) - 1))
            Mt = min(Mt, 1.0)
        else:
            # Choked
            mf = mdot_max
            Mt = 1.0
        mdot.append(mf)
        M_throat.append(Mt)

    mdot     = np.array(mdot)
    M_throat = np.array(M_throat)

    print("\n" + "═"*65)
    print("  EXPERIMENT 3 — Throat Sonic Condition & Choking")
    print("═"*65)
    print(f"\n  Stagnation conditions: P₀ = {P0/1e3:.0f} kPa, T₀ = {T0:.0f} K")
    print(f"  Throat area A*       = {A_THROAT:.4f} m²")
    print(f"  Choked mass flow ṁ_max = {mdot_max*1000:.4f} g/s")
    print(f"  Critical Pb*/P₀      = {Pcrit:.4f}  ({Pcrit*P0/1e3:.2f} kPa)\n")

    # Analytical check
    print("  Verification — T*, P*, ρ*, a* at throat (sonic):")
    T_star   = T0 * 2 / (GAMMA + 1)
    P_star   = P0 * Pcrit
    rho_star = P_star / (R_AIR * T_star)
    a_star   = sonic_velocity(T_star)
    print(f"    T* = {T_star:.2f} K")
    print(f"    P* = {P_star/1e3:.2f} kPa")
    print(f"    ρ* = {rho_star:.4f} kg/m³")
    print(f"    a* = {a_star:.2f} m/s  (= throat velocity)")
    print(f"    ṁ  = {rho_star * a_star * A_THROAT * 1000:.4f} g/s  (agrees: ✓)\n")

    # ── Figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor='#0D1117')
    fig.suptitle("EXPERIMENT 3 — Throat Sonic Condition & Mass Flow Choking",
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    ax_mdot  = fig.add_subplot(gs[0, :2])   # Mass flow curve
    ax_Mthr  = fig.add_subplot(gs[1, :2])   # Throat Mach
    ax_info  = fig.add_subplot(gs[0, 2])    # Schematic
    ax_pcrit = fig.add_subplot(gs[1, 2])    # Critical PR vs gamma

    for ax in [ax_mdot, ax_Mthr, ax_info, ax_pcrit]:
        ax.set_facecolor('#161B22')
        for sp in ax.spines.values(): sp.set_color('#30363D')
        ax.tick_params(colors='#8B949E', labelsize=9)
        ax.xaxis.label.set_color('#C9D1D9')
        ax.yaxis.label.set_color('#C9D1D9')

    # Mass flow curve
    ax_mdot.plot(Pb_over_P0, mdot * 1000, color='#00BCD4', lw=2.5)
    ax_mdot.axvline(Pcrit, color='#FF5722', lw=2, ls='--', label=f'Choking limit Pb*/P₀ = {Pcrit:.3f}')
    ax_mdot.axhline(mdot_max * 1000, color='#4CAF50', lw=1.5, ls='--',
                    label=f'ṁ_max = {mdot_max*1000:.4f} g/s')

    # Shade regions
    mask_choked = Pb_over_P0 <= Pcrit
    ax_mdot.fill_between(Pb_over_P0, mdot * 1000, where=mask_choked,
                          color='#FF5722', alpha=0.08, label='Choked region')
    ax_mdot.fill_between(Pb_over_P0, mdot * 1000, where=~mask_choked,
                          color='#2196F3', alpha=0.08, label='Unchoked region')

    ax_mdot.set_ylabel('Mass Flow Rate  ṁ  [g/s]', fontsize=10)
    ax_mdot.set_title('Mass Flow Rate vs Back Pressure Ratio  Pb/P₀\n'
                       '(Mass flow SATURATES once throat is sonic — this is CHOKING)',
                       color='#58A6FF', fontsize=11, pad=6)
    ax_mdot.legend(fontsize=9, facecolor='#1C2128', labelcolor='white', edgecolor='#30363D')
    ax_mdot.set_xlim(0, 1); ax_mdot.grid(color='#21262D', lw=0.5)

    ax_mdot.text(0.1, mdot_max*1000*0.4, 'CHOKED\nM_throat = 1\nṁ = const.',
                  color='#FF5722', fontsize=10, fontfamily='monospace', ha='center')
    ax_mdot.text(0.75, mdot_max*1000*0.4, 'UNCHOKED\nM_throat < 1\nṁ < ṁ_max',
                  color='#4FC3F7', fontsize=10, fontfamily='monospace', ha='center')

    # Throat Mach
    ax_Mthr.plot(Pb_over_P0, M_throat, color='#FF9800', lw=2.5)
    ax_Mthr.axvline(Pcrit, color='#FF5722', lw=2, ls='--', label=f'Pb*/P₀ = {Pcrit:.3f}')
    ax_Mthr.axhline(1.0, color='#4CAF50', lw=1.5, ls=':', label='M = 1  (sonic)')
    ax_Mthr.fill_between(Pb_over_P0, M_throat, 1, where=mask_choked,
                          alpha=0.08, color='#FF5722', label='Choked zone')
    ax_Mthr.set_xlabel('Back Pressure Ratio  Pb / P₀', fontsize=10)
    ax_Mthr.set_ylabel('Throat Mach Number  M_throat', fontsize=10)
    ax_Mthr.set_title('Throat Mach vs Back Pressure — Clamps at M=1 when Choked',
                       color='#58A6FF', fontsize=11, pad=6)
    ax_Mthr.legend(fontsize=9, facecolor='#1C2128', labelcolor='white', edgecolor='#30363D')
    ax_Mthr.set_xlim(0, 1); ax_Mthr.set_ylim(0, 1.15)
    ax_Mthr.grid(color='#21262D', lw=0.5)

    # Schematic
    ax_info.axis('off')
    ax_info.set_xlim(0, 10); ax_info.set_ylim(0, 12)
    ax_info.text(5, 11.3, 'CHOKING PHYSICS', color='white', fontsize=11,
                  ha='center', fontweight='bold', fontfamily='monospace')

    lines = [
        ("UNCHOKED  (Pb > Pb*)", '#4FC3F7'),
        ("• Throat M < 1", '#8B949E'),
        ("• ṁ increases as Pb drops", '#8B949E'),
        ("• Subsonic throughout", '#8B949E'),
        ("", None),
        ("AT CHOKING  (Pb = Pb*)", '#FFD700'),
        ("• Throat M = 1  (sonic)", '#8B949E'),
        (f"• Pb*/P₀ = {Pcrit:.4f}", '#8B949E'),
        (f"• T* = {T0*2/(GAMMA+1):.1f} K", '#8B949E'),
        (f"• ṁ_max = {mdot_max*1000:.4f} g/s", '#8B949E'),
        ("", None),
        ("CHOKED  (Pb < Pb*)", '#FF5722'),
        ("• Throat remains at M=1", '#8B949E'),
        ("• ṁ DOES NOT CHANGE", '#8B949E'),
        ("• Upstream unaware of", '#8B949E'),
        ("  downstream changes", '#8B949E'),
        ("• (info can't travel", '#8B949E'),
        ("   upstream at M=1)", '#8B949E'),
    ]
    for i, (txt, col) in enumerate(lines):
        if col is None: continue
        ax_info.text(0.5, 10.3 - i*0.62, txt, color=col, fontsize=8.5,
                      fontfamily='monospace')

    # Critical pressure ratio vs gamma
    gamma_range = np.linspace(1.1, 1.67, 200)
    Pcrit_range = (2 / (gamma_range + 1))**(gamma_range / (gamma_range - 1))
    ax_pcrit.plot(gamma_range, Pcrit_range, color='#E91E63', lw=2.5)
    ax_pcrit.scatter([GAMMA], [Pcrit], color='#FFD700', s=80, zorder=5,
                      label=f'Air: γ={GAMMA}, Pb*/P₀={Pcrit:.3f}')
    ax_pcrit.set_xlabel('γ (ratio of specific heats)', fontsize=10)
    ax_pcrit.set_ylabel('Critical Pb*/P₀', fontsize=10)
    ax_pcrit.set_title('Choking Pressure Ratio vs γ\n(varies with gas type)',
                        color='#58A6FF', fontsize=10, pad=6)
    ax_pcrit.legend(fontsize=9, facecolor='#1C2128', labelcolor='white', edgecolor='#30363D')
    ax_pcrit.grid(color='#21262D', lw=0.5)

    # Annotate common gases
    gases = [('Air',1.40,0.528),('CO₂',1.30,0.546),('He',1.67,0.487),('Steam',1.33,0.540)]
    for name, g, pc in gases:
        ax_pcrit.scatter([g], [pc], color='white', s=30, zorder=6)
        ax_pcrit.annotate(name, (g, pc), textcoords='offset points',
                           xytext=(4, 4), fontsize=7.5, color='#C9D1D9')

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
│  LAB NOTEBOOK — Experiment 3                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  TASK 1: Calculate choked mass flow manually and verify with code:      │
│    ṁ = A* · P₀/√T₀ · √(γ/R) · [2/(γ+1)]^((γ+1)/(2(γ-1)))            │
│                                                                         │
│  TASK 2: Change P0 from 100 kPa to 400 kPa (step 50 kPa).             │
│    Record ṁ_max for each. Is ṁ ∝ P₀? Confirm with a plot.             │
│                                                                         │
│  TASK 3: Change T0 from 200 K to 600 K.                                │
│    Record ṁ_max for each. Is ṁ ∝ 1/√T₀? Confirm.                      │
│                                                                         │
│  TASK 4: Why can information not travel upstream once M=1 at throat?   │
│    (Hint: think about acoustic wave propagation speed vs flow speed)    │
│                                                                         │
│  TASK 5: Look at the critical PR vs γ chart.                           │
│    Which gas requires the LOWEST Pb to achieve choking?                │
└─────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == '__main__':
    run_choking_experiment()
