"""
experiment_02_normal_shock.py
==============================
EXPERIMENT 2 — Normal Shock Location & Pressure Distribution
=============================================================

OBJECTIVE:
    Investigate how changing back pressure moves a normal shock through
    the diverging section, and quantify entropy/stagnation pressure losses.

THEORY:
    Across a normal shock (Rankine-Hugoniot relations):
        M₂²  = [(γ-1)M₁² + 2] / [2γM₁² - (γ-1)]
        P₂/P₁ = 1 + 2γ/(γ+1) * (M₁² - 1)
        T₂/T₁ = P₂/P₁ * (2 + (γ-1)M₁²) / ((γ+1)M₁²)
        P₀₂/P₀₁ = [(γ+1)M₁²/(2+(γ-1)M₁²)]^(γ/(γ-1)) * [2γM₁²/(γ+1) - (γ-1)/(γ+1)]^(-1/(γ-1))

    The shock moves toward the exit as back pressure decreases.
    At design condition: no shock exists anywhere in/outside nozzle.

STUDENT TASKS (see bottom of file):
    ① Observe shock jump in M and P plots
    ② Compute entropy generation across shock
    ③ Plot P₀ loss vs shock Mach number

RUN:
    python experiment_02_normal_shock.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nozzle_physics import (NozzleFlowSolver, normal_shock_M2,
                             normal_shock_P_ratio, normal_shock_T_ratio,
                             normal_shock_P0_ratio, mach_from_area_ratio,
                             isentropic_P_ratio, R_AIR)

# ══════════════════════════════════════════════════════════════════════════════
#  ██ STUDENT-ADJUSTABLE PARAMETERS  ██
# ══════════════════════════════════════════════════════════════════════════════

P0      = 300_000   # Stagnation pressure [Pa]
T0      = 350.0     # Stagnation temperature [K]
AR_exit = 3.5       # Exit area ratio A_e/A*

# Back pressures [Pa] — must be between Pb_supersonic and Pb_shock_at_exit
# Leave as None to auto-generate 5 sweep cases
CUSTOM_Pb_LIST = None   # e.g., [150e3, 180e3, 210e3, 240e3]

# ══════════════════════════════════════════════════════════════════════════════

def shock_sweep(P0=300_000, T0=350.0, AR_exit=3.5, CUSTOM_Pb_LIST=None):
    import io, base64
    solver = NozzleFlowSolver(P0=P0, T0=T0, AR_exit=AR_exit, n_pts=600)

    if CUSTOM_Pb_LIST is not None:
        pb_list = CUSTOM_Pb_LIST
    else:
        # Auto: sweep from "shock just past throat" to "shock at exit"
        pb_list = np.linspace(solver.Pb_shock_at_exit * 0.98,
                              solver.Pb_subsonic * 0.92, 5)

    print("\n" + "═"*65)
    print(f"  EXPERIMENT 2 — Normal Shock Analysis")
    print(f"  P₀={P0/1e3:.0f} kPa  |  T₀={T0:.0f} K  |  AR_exit={AR_exit}")
    print("═"*65)
    print(f"\n  Critical back pressures (Pa):")
    print(f"    Pb_subsonic_isen  = {solver.Pb_subsonic/1e3:.2f} kPa  (no shock, unchoked)")
    print(f"    Pb_shock_at_exit  = {solver.Pb_shock_at_exit/1e3:.2f} kPa  (shock right at exit)")
    print(f"    Pb_design         = {solver.Pb_supersonic/1e3:.2f} kPa  (fully supersonic)")
    print(f"    M_design (exit)   = {solver.M_exit_design:.4f}\n")

    fig = plt.figure(figsize=(18, 14), facecolor='#0D1117')
    fig.suptitle("EXPERIMENT 2 — Normal Shock in C-D Nozzle\nPressure & Mach Number Jump Analysis",
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    ax_M   = fig.add_subplot(gs[0, :2])
    ax_P   = fig.add_subplot(gs[1, :2])
    ax_s   = fig.add_subplot(gs[2, 0])
    ax_P0  = fig.add_subplot(gs[2, 1])
    ax_nr  = fig.add_subplot(gs[0:2, 2])
    ax_tbl = fig.add_subplot(gs[2, 2])

    dark_style = dict(facecolor='#161B22')
    for ax in [ax_M, ax_P, ax_s, ax_P0, ax_nr, ax_tbl]:
        ax.set_facecolor('#161B22')
        for sp in ax.spines.values(): sp.set_color('#30363D')
        ax.tick_params(colors='#8B949E', labelsize=9)
        ax.xaxis.label.set_color('#C9D1D9')
        ax.yaxis.label.set_color('#C9D1D9')

    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(pb_list)-1, 1)) for i in range(len(pb_list))]

    shock_records = []

    for idx, Pb in enumerate(pb_list):
        sol  = solver.solve(Pb)
        col  = colors[idx]
        lbl  = f"Pb = {Pb/1e3:.1f} kPa"

        ax_M.plot(sol['x'], sol['M'],  color=col, lw=1.8, label=lbl)
        ax_P.plot(sol['x'], sol['P']/1e3, color=col, lw=1.8, label=lbl)

        if sol['shock_x'] is not None:
            si = sol['shock_idx']
            M1 = sol['M'][si-1]
            M2 = normal_shock_M2(M1)
            P0_loss = 1 - normal_shock_P0_ratio(M1)
            ds_over_R = -np.log(normal_shock_P0_ratio(M1))  # Δs/R = -ln(P02/P01)

            ax_M.axvline(sol['shock_x'], color=col, lw=0.8, ls=':', alpha=0.5)
            ax_P.axvline(sol['shock_x'], color=col, lw=0.8, ls=':', alpha=0.5)

            shock_records.append({
                'Pb_kPa' : Pb/1e3,
                'x_shock': sol['shock_x'],
                'M1'     : M1,
                'M2'     : M2,
                'P2/P1'  : normal_shock_P_ratio(M1),
                'T2/T1'  : normal_shock_T_ratio(M1),
                'P02/P01': normal_shock_P0_ratio(M1),
                'Ds/R'   : ds_over_R,
                'col'    : col,
            })
            print(f"  Pb={Pb/1e3:6.1f} kPa  |  x_shock={sol['shock_x']:.3f}  "
                  f"|  M1={M1:.3f}  M2={M2:.3f}  "
                  f"|  P02/P01={normal_shock_P0_ratio(M1):.4f}  "
                  f"|  Δs/R={ds_over_R:.4f}")

    # Annotations
    for ax in [ax_M, ax_P]:
        ti = solver.throat_idx
        ax.axvline(solver.x[ti], color='#FFD700', lw=1, ls='--', alpha=0.6)
        ax.axhline(1.0 if ax is ax_M else (P0/isentropic_P_ratio(1.0))/1e3,
                   color='#FF6B6B', lw=1, ls=':', alpha=0.4)

    ax_M.set_ylabel('Mach Number  M', fontsize=10)
    ax_M.set_title('Mach Number — Shock Sweeps',  color='#58A6FF', fontsize=11, pad=6)
    ax_M.legend(fontsize=8, facecolor='#1C2128', labelcolor='white',
                edgecolor='#30363D', ncol=2)
    ax_M.set_xlim(0, 1); ax_M.grid(color='#21262D', lw=0.5)

    ax_P.set_xlabel('Non-dim. axial position  x/L', fontsize=10)
    ax_P.set_ylabel('Static Pressure  P  [kPa]', fontsize=10)
    ax_P.set_title('Pressure Distribution — Shock Jump Visible', color='#58A6FF', fontsize=11, pad=6)
    ax_P.legend(fontsize=8, facecolor='#1C2128', labelcolor='white',
                edgecolor='#30363D', ncol=2)
    ax_P.set_xlim(0, 1); ax_P.grid(color='#21262D', lw=0.5)

    # ── Entropy increase vs M1 ────────────────────────────────────────────
    M1_range = np.linspace(1.001, 5.0, 300)
    ds_range = [-np.log(normal_shock_P0_ratio(m)) for m in M1_range]
    ax_s.plot(M1_range, ds_range, color='#FF5722', lw=2)
    ax_s.set_xlabel('Upstream Mach  M₁', fontsize=10)
    ax_s.set_ylabel('Δs / R  (entropy generation)', fontsize=9)
    ax_s.set_title('Entropy Rise vs Shock Strength',
                   color='#58A6FF', fontsize=10, pad=6)
    ax_s.grid(color='#21262D', lw=0.5)
    for r in shock_records:
        ax_s.scatter(r['M1'], r['Ds/R'], color=r['col'], zorder=5, s=60)
        ax_s.annotate(f"Pb={r['Pb_kPa']:.0f}", (r['M1'], r['Ds/R']),
                       textcoords='offset points', xytext=(5, 2),
                       fontsize=7, color=r['col'])

    # ── P02/P01 vs M1 ────────────────────────────────────────────────────
    P0_ratio = [normal_shock_P0_ratio(m) for m in M1_range]
    ax_P0.plot(M1_range, P0_ratio, color='#2196F3', lw=2)
    ax_P0.set_xlabel('Upstream Mach  M₁', fontsize=10)
    ax_P0.set_ylabel('P₀₂/P₀₁  (stagnation pressure ratio)', fontsize=8)
    ax_P0.set_title('Stagnation Pressure Loss\nacross Normal Shock',
                    color='#58A6FF', fontsize=10, pad=6)
    ax_P0.axhline(1.0, color='#4CAF50', lw=1, ls='--', alpha=0.5, label='No loss (ideal)')
    ax_P0.grid(color='#21262D', lw=0.5)
    ax_P0.legend(fontsize=8, facecolor='#1C2128', labelcolor='white', edgecolor='#30363D')
    for r in shock_records:
        ax_P0.scatter(r['M1'], r['P02/P01'], color=r['col'], zorder=5, s=60)

    # ── Normal shock schematic ────────────────────────────────────────────
    ax_nr.axis('off')
    ax_nr.set_xlim(0, 10)
    ax_nr.set_ylim(0, 12)

    # Draw schematic
    ax_nr.add_patch(Rectangle((0.5, 5), 3.5, 2, color='#1E3A5F', zorder=2))
    ax_nr.add_patch(Rectangle((6,   5), 3.5, 2, color='#3A1E1E', zorder=2))
    ax_nr.axvline(5, color='#FF5722', lw=3, ymin=0.35, ymax=0.65, zorder=3)

    ax_nr.text(2, 9.5, 'NORMAL SHOCK DIAGRAM', color='white', fontsize=11,
               ha='center', fontweight='bold', fontfamily='monospace')

    # Arrows
    for y in [5.5, 6, 6.5]:
        ax_nr.annotate('', xy=(4.8, y), xytext=(0.7, y),
                        arrowprops=dict(arrowstyle='->', color='#4FC3F7', lw=1.5))
    for y in [5.5, 6, 6.5]:
        ax_nr.annotate('', xy=(9.3, y), xytext=(5.2, y),
                        arrowprops=dict(arrowstyle='->', color='#EF9A9A', lw=1.5))

    ax_nr.text(2.3, 7.5, 'M₁ > 1  (Supersonic)', color='#4FC3F7', fontsize=10, ha='center')
    ax_nr.text(7.7, 7.5, 'M₂ < 1  (Subsonic)',   color='#EF9A9A', fontsize=10, ha='center')
    ax_nr.text(5,   4.2, 'SHOCK\nFRONT', color='#FF5722', fontsize=9, ha='center',
               fontfamily='monospace')

    props = [
        ('P₂/P₁', 'pressure increases'),
        ('T₂/T₁', 'temperature increases'),
        ('ρ₂/ρ₁', 'density increases'),
        ('P₀₂/P₀₁', 'stagnation P drops'),
        ('Δs > 0', 'irreversible → entropy ↑'),
    ]
    for i, (sym, desc) in enumerate(props):
        ax_nr.text(1, 3.4 - i*0.65, f'• {sym}:', color='#FFD700', fontsize=9,
                   fontfamily='monospace')
        ax_nr.text(4.2, 3.4 - i*0.65, desc, color='#8B949E', fontsize=9)

    if shock_records:
        r = shock_records[len(shock_records)//2]
        vals = [
            f"M₁     = {r['M1']:.3f}",
            f"M₂     = {r['M2']:.3f}",
            f"P₂/P₁  = {r['P2/P1']:.3f}",
            f"T₂/T₁  = {r['T2/T1']:.3f}",
            f"P₀₂/P₀₁= {r['P02/P01']:.4f}",
            f"Δs/R   = {r['Ds/R']:.4f}",
        ]
        ax_nr.text(5.5, 3.9, f'Mid-sweep\n(Pb={r["Pb_kPa"]:.0f} kPa)', color='#58A6FF',
                   fontsize=9, fontfamily='monospace')
        for i, v in enumerate(vals):
            ax_nr.text(5.5, 3.1 - i*0.6, v, color='#C9D1D9', fontsize=8.5,
                       fontfamily='monospace')

    # ── Results table ─────────────────────────────────────────────────────
    ax_tbl.axis('off')
    if shock_records:
        rows = [[f"{r['Pb_kPa']:.1f}",
                 f"{r['x_shock']:.3f}",
                 f"{r['M1']:.3f}",
                 f"{r['M2']:.3f}",
                 f"{r['P02/P01']:.4f}",
                 f"{r['Ds/R']:.4f}"]
                for r in shock_records]
        cols = ['Pb\n(kPa)','x_shock','M₁','M₂','P₀₂/P₀₁','Δs/R']
        tbl = ax_tbl.table(cellText=rows, colLabels=cols,
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#1C2128' if r > 0 else '#21262D')
            cell.set_text_props(color='#C9D1D9' if r > 0 else '#58A6FF',
                                fontfamily='monospace')
            cell.set_edgecolor('#30363D')
        ax_tbl.set_title('Shock Jump Data Table', color='#58A6FF', fontsize=10, pad=8)

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
│  LAB NOTEBOOK — Experiment 2                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  TASK 1: Fill in the Shock Jump Data Table in your notebook.            │
│    Plot shock position x_shock vs Pb on graph paper.                   │
│    Trend: Does shock move toward or away from exit as Pb decreases?    │
│                                                                         │
│  TASK 2: Calculate entropy increase for each shock case:               │
│    Δs = -R * ln(P02/P01)   [J/(kg·K)]                                  │
│    Confirm using the Ds/R column × R_air (287 J/kg·K)                  │
│                                                                         │
│  TASK 3: For M1 = 2.0, manually compute M2, P2/P1, T2/T1              │
│    using the Rankine-Hugoniot equations. Verify with the plot.          │
│                                                                         │
│  TASK 4: Why is P02/P01 always < 1 across a shock?                     │
│    Relate this to the 2nd Law of Thermodynamics.                       │
│                                                                         │
│  TASK 5: Modify CUSTOM_Pb_LIST to place shock exactly at x=0.7.        │
│    (Hint: use binary search with Pb between the two critical values)    │
└─────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == '__main__':
    shock_sweep()
