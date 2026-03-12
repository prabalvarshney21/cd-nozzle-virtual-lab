"""
experiment_01_mach_distribution.py
====================================
EXPERIMENT 1 — Mach Number Distribution & Color Map
====================================================

OBJECTIVE:
    Visualise how the Mach number varies along a convergent-divergent nozzle
    for different back-pressure conditions (flow regimes).

THEORY:
    In a C-D nozzle, the area-Mach relationship (A/A* relation) governs
    local Mach number for isentropic flow:

        A/A* = (1/M) * [(2/(γ+1)) * (1 + (γ-1)/2 * M²)]^((γ+1)/(2(γ-1)))

    Four distinct operating regimes exist:
      1. Subsonic (unchoked)           — everywhere M < 1
      2. Shock in diverging section    — M = 1 at throat, normal shock inside
      3. Isentropic supersonic         — design condition, M > 1 in diverging
      4. Over/under-expanded           — external shocks outside nozzle

STUDENT TASKS (see bottom of file):
    ① Change AR_exit and observe how design Mach number shifts
    ② Change back pressure Pb and observe shock movement
    ③ Record Mach exit values in your lab notebook table

RUN:
    python experiment_01_mach_distribution.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nozzle_physics import NozzleFlowSolver, mach_from_area_ratio, area_mach_ratio

# ══════════════════════════════════════════════════════════════════════════════
#  ██ STUDENT-ADJUSTABLE PARAMETERS  ██
# ══════════════════════════════════════════════════════════════════════════════

P0      = 200_000   # Reservoir / stagnation pressure [Pa]   (try 100k–500k)
T0      = 300.0     # Stagnation temperature [K]             (try 200–800)
AR_exit = 3.0       # Exit-to-throat area ratio A_e/A*       (try 2–6)

# Back pressures to compare  [Pa]  — defines different regimes
Pb_cases = {
    "Subsonic (unchoked)"       : P0 * 0.95,
    "Shock near exit"           : P0 * 0.40,
    "Shock mid-diverge"         : P0 * 0.55,
    "Design (fully supersonic)" : None,        # None → isentropic supersonic
}

# ══════════════════════════════════════════════════════════════════════════════

def compute_design_Pb(solver):
    """Compute exact back pressure for fully supersonic (design) condition."""
    from nozzle_physics import isentropic_P_ratio
    M_e = mach_from_area_ratio(solver.AR_exit, supersonic=True)
    return solver.P0 / isentropic_P_ratio(M_e)


def plot_mach_distribution(P0=200_000, T0=300.0, AR_exit=3.0,
                            pb_ratio_subsonic=0.95, pb_ratio_shock_near=0.40,
                            pb_ratio_shock_mid=0.55):
    import io, base64
    pb_cases = {
        "Subsonic (unchoked)"       : P0 * pb_ratio_subsonic,
        "Shock near exit"           : P0 * pb_ratio_shock_near,
        "Shock mid-diverge"         : P0 * pb_ratio_shock_mid,
        "Design (fully supersonic)" : None,
    }
    solver = NozzleFlowSolver(P0=P0, T0=T0, AR_exit=AR_exit, n_pts=600)

    # Fill in the design Pb
    pb_cases["Design (fully supersonic)"] = compute_design_Pb(solver)

    colors = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50']
    lw     = [1.8, 1.8, 2.2, 2.5]
    ls     = ['--', '-.', ':', '-']

    fig = plt.figure(figsize=(16, 12), facecolor='#0D1117')
    fig.suptitle("EXPERIMENT 1 — Mach Number Distribution\nConvex-Divergent Nozzle (C-D Nozzle)",
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.93, bottom=0.05)

    ax_geo  = fig.add_subplot(gs[0, :])    # Nozzle geometry
    ax_mach = fig.add_subplot(gs[1, :])    # Mach profiles
    ax_Aratio = fig.add_subplot(gs[2, 0])  # A/A* curve
    ax_table  = fig.add_subplot(gs[2, 1])  # Results table

    for ax in [ax_geo, ax_mach, ax_Aratio, ax_table]:
        ax.set_facecolor('#161B22')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
        ax.tick_params(colors='#8B949E', labelsize=9)
        ax.xaxis.label.set_color('#C9D1D9')
        ax.yaxis.label.set_color('#C9D1D9')

    # ── Plot 1: Nozzle geometry cross-section ─────────────────────────────
    x = solver.x
    AR = solver.AR
    # Half-height proportional to sqrt(A/A*)
    half_h = np.sqrt(AR)
    half_h_norm = half_h / half_h.max() * 0.45

    ax_geo.fill_between(x,  half_h_norm, 1.0, color='#1F2937', alpha=0.9)
    ax_geo.fill_between(x, -half_h_norm,-1.0, color='#1F2937', alpha=0.9)
    ax_geo.fill_between(x,  half_h_norm, -half_h_norm,
                         color='#0D2137', alpha=0.3, label='Flow area')

    # Colour the flow by Mach (design condition)
    sol_design = solver.solve(compute_design_Pb(solver))
    M_des = sol_design['M']
    cmap_mach = cm.jet
    norm_m = Normalize(vmin=0, vmax=M_des.max())

    for i in range(len(x) - 1):
        ax_geo.fill_betweenx([-half_h_norm[i], half_h_norm[i]],
                              x[i], x[i+1],
                              color=cmap_mach(norm_m(M_des[i])), alpha=0.85)

    ax_geo.plot(x,  half_h_norm, color='#58A6FF', lw=2.5)
    ax_geo.plot(x, -half_h_norm, color='#58A6FF', lw=2.5)

    # Throat marker
    ti = solver.throat_idx
    ax_geo.axvline(x[ti], color='#FFD700', lw=1.5, ls='--', alpha=0.8)
    ax_geo.text(x[ti]+0.01, 0.6, f'THROAT\nA/A*=1', color='#FFD700',
                fontsize=8, va='center', fontfamily='monospace')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_mach, norm=norm_m)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_geo, orientation='vertical', fraction=0.015, pad=0.01)
    cbar.set_label('Mach Number', color='#C9D1D9', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8B949E', fontsize=8)

    ax_geo.set_xlim(0, 1)
    ax_geo.set_ylim(-1.05, 1.05)
    ax_geo.set_xlabel('Non-dimensional Axial Position  x/L', fontsize=10)
    ax_geo.set_ylabel('Height (norm.)', fontsize=10)
    ax_geo.set_title('Nozzle Cross-Section — Mach Number Color Map (Design Condition)',
                     color='#58A6FF', fontsize=11, pad=8)
    ax_geo.text(0.02, 0.65, f'A_e/A* = {AR_exit}', color='white', fontsize=9,
                transform=ax_geo.transAxes, fontfamily='monospace')
    ax_geo.text(0.02, 0.55, f'P₀ = {P0/1e3:.0f} kPa', color='white', fontsize=9,
                transform=ax_geo.transAxes, fontfamily='monospace')
    ax_geo.text(0.02, 0.45, f'T₀ = {T0:.0f} K', color='white', fontsize=9,
                transform=ax_geo.transAxes, fontfamily='monospace')

    # ── Plot 2: Mach number profiles for each case ────────────────────────
    results_table = []
    for idx, (label, Pb) in enumerate(pb_cases.items()):
        sol = solver.solve(Pb)
        ax_mach.plot(sol['x'], sol['M'],
                     color=colors[idx], lw=lw[idx], ls=ls[idx],
                     label=f"{label}  (Pb={Pb/1e3:.1f} kPa)")

        # Shock indicator
        if sol['shock_x'] is not None:
            ax_mach.axvline(sol['shock_x'], color=colors[idx],
                            lw=1, ls=':', alpha=0.5)
            ax_mach.annotate('⚡ Shock', xy=(sol['shock_x'], sol['M'][sol['shock_idx']]),
                              xytext=(sol['shock_x']+0.05, sol['M'][sol['shock_idx']]+0.3),
                              color=colors[idx], fontsize=8,
                              arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1))

        results_table.append([label,
                               f"{Pb/1e3:.1f}",
                               f"{sol['M_exit']:.4f}",
                               f"{sol['P_exit']/1e3:.2f}",
                               f"{sol['T_exit']:.1f}",
                               sol['regime'][:30]])

    ax_mach.axvline(x[ti], color='#FFD700', lw=1, ls='--', alpha=0.6,
                    label='Throat (x/L = {:.2f})'.format(x[ti]))
    ax_mach.axhline(1.0, color='#FF6B6B', lw=1, ls=':', alpha=0.5, label='M = 1 (sonic)')

    ax_mach.set_xlabel('Non-dimensional Axial Position  x/L', fontsize=10)
    ax_mach.set_ylabel('Mach Number  M', fontsize=10)
    ax_mach.set_title('Mach Number Distribution — Four Flow Regimes',
                      color='#58A6FF', fontsize=11, pad=8)
    ax_mach.legend(fontsize=8, facecolor='#1C2128', labelcolor='white',
                   edgecolor='#30363D', loc='upper left')
    ax_mach.set_xlim(0, 1)
    ax_mach.set_ylim(-0.05, sol_design['M'].max() * 1.15)
    ax_mach.grid(color='#21262D', lw=0.5)

    # Region labels
    ax_mach.text(x[ti]/2, 0.05, 'CONVERGING\nSECTION',
                 color='#6E7681', fontsize=8, ha='center', va='bottom',
                 fontfamily='monospace')
    ax_mach.text((x[ti]+1)/2, 0.05, 'DIVERGING\nSECTION',
                 color='#6E7681', fontsize=8, ha='center', va='bottom',
                 fontfamily='monospace')

    # ── Plot 3: A/A* curve with Mach mapping ─────────────────────────────
    M_range_sub = np.linspace(0.01, 1.0, 300)
    M_range_sup = np.linspace(1.0, 4.5, 300)
    AR_sub = np.array([area_mach_ratio(m) for m in M_range_sub])
    AR_sup = np.array([area_mach_ratio(m) for m in M_range_sup])

    ax_Aratio.plot(AR_sub, M_range_sub, color='#2196F3', lw=2, label='Subsonic branch')
    ax_Aratio.plot(AR_sup, M_range_sup, color='#FF5722', lw=2, label='Supersonic branch')
    ax_Aratio.axhline(1.0, color='#FFD700', lw=1, ls='--', alpha=0.7, label='Sonic (M=1)')
    ax_Aratio.axvline(AR_exit, color='#4CAF50', lw=1.5, ls='--', alpha=0.7,
                      label=f'Design AR = {AR_exit}')

    M_sub_design = mach_from_area_ratio(AR_exit, supersonic=False)
    M_sup_design = mach_from_area_ratio(AR_exit, supersonic=True)
    ax_Aratio.scatter([AR_exit, AR_exit], [M_sub_design, M_sup_design],
                      color=['#2196F3','#FF5722'], zorder=5, s=60)
    ax_Aratio.annotate(f'M={M_sub_design:.3f}', (AR_exit, M_sub_design),
                       xytext=(AR_exit+0.2, M_sub_design-0.2),
                       color='#2196F3', fontsize=8, arrowprops=dict(arrowstyle='->', color='#2196F3'))
    ax_Aratio.annotate(f'M={M_sup_design:.3f}', (AR_exit, M_sup_design),
                       xytext=(AR_exit+0.2, M_sup_design+0.2),
                       color='#FF5722', fontsize=8, arrowprops=dict(arrowstyle='->', color='#FF5722'))

    ax_Aratio.set_xlabel('Area Ratio  A/A*', fontsize=10)
    ax_Aratio.set_ylabel('Mach Number  M', fontsize=10)
    ax_Aratio.set_title('A/A* — Mach Relation\n(Both solution branches)',
                        color='#58A6FF', fontsize=10, pad=6)
    ax_Aratio.legend(fontsize=7.5, facecolor='#1C2128', labelcolor='white',
                     edgecolor='#30363D')
    ax_Aratio.set_xlim(1, 6)
    ax_Aratio.set_ylim(0, 4.5)
    ax_Aratio.grid(color='#21262D', lw=0.5)

    # ── Plot 4: Results table ─────────────────────────────────────────────
    ax_table.axis('off')
    col_labels = ['Case', 'Pb\n(kPa)', 'M_exit', 'P_exit\n(kPa)', 'T_exit\n(K)', 'Regime']
    tbl = ax_table.table(cellText=results_table,
                         colLabels=col_labels,
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1C2128' if r > 0 else '#21262D')
        cell.set_text_props(color='#C9D1D9' if r > 0 else '#58A6FF',
                            fontfamily='monospace')
        cell.set_edgecolor('#30363D')
    ax_table.set_title('Exit Conditions Summary  →  Fill in your Lab Notebook',
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
│  LAB NOTEBOOK — Experiment 1                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  TASK 1: Vary AR_exit (line 45).                                        │
│    Try AR_exit = 2.0, 3.0, 4.0, 5.0                                    │
│    Record M_exit (design) for each → plot M_design vs AR_exit           │
│                                                                         │
│  TASK 2: For AR_exit = 3.0, change P0 from 100 kPa to 400 kPa.        │
│    Does M_exit change? Why or why not?                                  │
│                                                                         │
│  TASK 3: In the Mach color map, identify:                               │
│    a) High-pressure stagnation zone (inlet)                             │
│    b) Sonic throat (M = 1)                                              │
│    c) Supersonic expansion zone                                         │
│    Sketch and label in your notebook.                                   │
│                                                                         │
│  TASK 4: Why does the A/A* curve have TWO branches?                    │
│    Explain physically what determines which branch the flow follows.    │
└─────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == '__main__':
    plot_mach_distribution()
