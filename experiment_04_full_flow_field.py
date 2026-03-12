"""
experiment_04_full_flow_field.py
==================================
EXPERIMENT 4 — Full Flow Field: Temperature, Pressure, Density, Velocity
=========================================================================

OBJECTIVE:
    Visualise ALL five flow properties simultaneously at the design
    (fully supersonic) condition and compare with a shock-in-nozzle case.
    Understand how each property couples through isentropic relations.

THEORY:
    For isentropic flow:
        T/T₀   = [1 + (γ-1)/2 · M²]⁻¹
        P/P₀   = [1 + (γ-1)/2 · M²]^(-γ/(γ-1))
        ρ/ρ₀   = [1 + (γ-1)/2 · M²]^(-1/(γ-1))
        V = M · a = M · √(γRT)

    Mass conservation: ρ·V·A = const  →  as A decreases → ρV must increase

STUDENT TASKS (see bottom of file):
    ① Verify isentropic relations at exit using printed values
    ② Compare design vs shock condition side-by-side
    ③ Compute thrust using momentum equation

RUN:
    python experiment_04_full_flow_field.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nozzle_physics import NozzleFlowSolver, R_AIR, GAMMA_AIR

# ══════════════════════════════════════════════════════════════════════════════
#  ██ STUDENT-ADJUSTABLE PARAMETERS  ██
# ══════════════════════════════════════════════════════════════════════════════

P0      = 500_000    # Stagnation pressure [Pa]     (try 100k–1000k)
T0      = 500.0      # Stagnation temperature [K]   (try 300–1000)
AR_exit = 4.0        # Exit area ratio               (try 2–8)
A_THROAT= 0.005      # Throat area [m²]

# Choose comparison case: 'shock_mid', 'subsonic', or None (no comparison)
COMPARISON = 'shock_mid'

# ══════════════════════════════════════════════════════════════════════════════

def plot_2d_field(ax, x, AR, field, cmap, title, unit, colorbar_ax=None):
    """
    Plot a 2-D symmetric nozzle cross-section colored by a field variable.
    """
    # Half-height = sqrt(A/A*) normalised
    half_h = np.sqrt(AR) / np.sqrt(AR.max()) * 0.45

    # Build 2-D grid
    ny = 40
    Y  = np.linspace(0, 1, ny)   # 0=centre, 1=wall

    Z = np.zeros((ny, len(x)))
    for j in range(len(x)):
        Z[:, j] = field[j]  # uniform across cross-section (1-D approx)

    X2, Y2 = np.meshgrid(x, Y)
    # Mask outside nozzle
    for j in range(len(x)):
        mask = Y > half_h[j] / 0.45
        Z[mask, j] = np.nan

    im = ax.pcolormesh(X2, Y2, Z, cmap=cmap, shading='auto')

    # Mirror for bottom half
    Y2b = -Y2
    im2 = ax.pcolormesh(X2, Y2b, Z, cmap=cmap, shading='auto')

    # Nozzle walls
    ax.plot(x,  half_h / 0.45, color='#58A6FF', lw=2)
    ax.plot(x, -half_h / 0.45, color='#58A6FF', lw=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(title, color='#58A6FF', fontsize=10, pad=4)
    ax.set_facecolor('#0D1117')

    return im


def run_flow_field(P0=500_000, T0=500.0, AR_exit=4.0, A_THROAT=0.005, COMPARISON='shock_mid'):
    import io, base64
    solver = NozzleFlowSolver(P0=P0, T0=T0, AR_exit=AR_exit, n_pts=400)

    # Design condition
    sol_d = solver.solve(solver.Pb_supersonic)

    # Comparison condition
    if COMPARISON == 'shock_mid':
        Pb_comp = (solver.Pb_shock_at_exit + solver.Pb_subsonic) * 0.5
        sol_c   = solver.solve(Pb_comp)
        comp_label = f"Normal Shock (Pb={Pb_comp/1e3:.0f} kPa)"
    elif COMPARISON == 'subsonic':
        sol_c = solver.solve(solver.Pb_subsonic * 0.98)
        comp_label = "Subsonic (unchoked)"
    else:
        sol_c = None

    print("\n" + "═"*65)
    print("  EXPERIMENT 4 — Full Flow Field Visualization")
    print("═"*65)
    print(f"\n  Stagnation: P₀={P0/1e3:.0f} kPa, T₀={T0:.0f} K, AR_exit={AR_exit}")
    print(f"\n  DESIGN CONDITION (isentropic supersonic):")
    print(f"    M_exit    = {sol_d['M_exit']:.4f}")
    print(f"    P_exit    = {sol_d['P_exit']/1e3:.3f} kPa  (P_exit/P₀ = {sol_d['P_exit']/P0:.5f})")
    print(f"    T_exit    = {sol_d['T_exit']:.2f} K  (T_exit/T₀ = {sol_d['T_exit']/T0:.5f})")
    print(f"    V_exit    = {sol_d['V'][-1]:.2f} m/s")
    rho_exit = sol_d['rho'][-1]
    print(f"    ρ_exit    = {rho_exit:.5f} kg/m³")
    thrust = rho_exit * sol_d['V'][-1]**2 * A_THROAT * AR_exit
    print(f"    Thrust F  = ρ·V²·A_exit = {thrust:.2f} N  (gross, no ambient subtracted)")

    # ── Figure layout ─────────────────────────────────────────────────────
    ncols = 2 if sol_c else 1
    fig = plt.figure(figsize=(18 if sol_c else 14, 16), facecolor='#0D1117')
    fig.suptitle("EXPERIMENT 4 — Full Flow Field: P, T, ρ, V, M\nDesign Condition vs Comparison",
                 fontsize=15, color='white', fontweight='bold', y=0.99)

    props = [
        ('M',   'Mach Number',          'jet',         None),
        ('P',   'Static Pressure [Pa]', 'plasma',      None),
        ('T',   'Temperature [K]',      'hot',         None),
        ('rho', 'Density [kg/m³]',      'YlOrRd',      None),
        ('V',   'Velocity [m/s]',       'cool',        None),
    ]

    n_props = len(props)
    rows = n_props
    gs = fig.add_gridspec(rows, ncols * 2 + 1,
                          hspace=0.55, wspace=0.1,
                          left=0.04, right=0.97,
                          top=0.95, bottom=0.04)

    solutions = [sol_d]
    labels    = [f"DESIGN CONDITION\n(fully supersonic, Pb={solver.Pb_supersonic/1e3:.1f} kPa)"]
    if sol_c:
        solutions.append(sol_c)
        labels.append(comp_label)

    for pi, (key, name, cmap_name, _) in enumerate(props):
        for si, (sol, lbl) in enumerate(zip(solutions, labels)):
            col_start = si * 2
            ax = fig.add_subplot(gs[pi, col_start:col_start+2])
            for sp in ax.spines.values(): sp.set_color('#30363D')
            ax.tick_params(colors='#8B949E', labelsize=7)

            field = sol[key]
            vmin, vmax = field.min(), field.max()

            im = plot_2d_field(ax, sol['x'], sol['AR'], field,
                               cmap_name,
                               f"{'← ' if si==0 else ''}{name}  |  {lbl[:40]}",
                               name)

            im.set_clim(vmin, vmax)

            # Colorbar in last column
            if si == len(solutions) - 1:
                cax = fig.add_subplot(gs[pi, ncols * 2])
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(name, color='#C9D1D9', fontsize=8)
                cbar.ax.yaxis.set_tick_params(color='#8B949E')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8B949E', fontsize=7)

            # Throat marker
            ti = solver.throat_idx
            ax.axvline(sol['x'][ti], color='#FFD700', lw=1, ls='--', alpha=0.6)

            # Shock marker
            if sol['shock_x'] is not None:
                ax.axvline(sol['shock_x'], color='#FF5722', lw=1.5, ls='-', alpha=0.8)
                if pi == 0:
                    ax.text(sol['shock_x']+0.01, 0.5, '⚡',
                            color='#FF5722', fontsize=10, va='center')

            if pi == rows - 1:
                ax.set_xlabel('x/L', fontsize=8, color='#C9D1D9')

    # Property profiles comparison (1-D line plots)
    fig2, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor='#0D1117')
    fig2.suptitle("EXPERIMENT 4 — 1-D Property Profiles Comparison",
                   fontsize=14, color='white', fontweight='bold')

    plot_keys = [('M','Mach Number',''),
                 ('P','Static Pressure','Pa'),
                 ('T','Temperature','K'),
                 ('rho','Density','kg/m³'),
                 ('V','Velocity','m/s')]

    sol_colors = ['#00BCD4','#FF9800']
    for idx, (ax, (key, name, unit)) in enumerate(zip(axes.flat[:5], plot_keys)):
        ax.set_facecolor('#161B22')
        for sp in ax.spines.values(): sp.set_color('#30363D')
        ax.tick_params(colors='#8B949E', labelsize=9)
        ax.xaxis.label.set_color('#C9D1D9')
        ax.yaxis.label.set_color('#C9D1D9')

        for si, (sol, lbl, col) in enumerate(zip(solutions, labels, sol_colors)):
            ax.plot(sol['x'], sol[key], color=col, lw=2,
                    label=lbl[:35], ls='-' if si == 0 else '--')
            if sol['shock_x'] is not None:
                ax.axvline(sol['shock_x'], color=col, lw=1, ls=':', alpha=0.5)

        ax.axvline(solver.x[solver.throat_idx], color='#FFD700',
                   lw=1, ls='--', alpha=0.5, label='Throat')
        ax.set_xlabel('x/L', fontsize=9)
        ax.set_ylabel(f'{name} [{unit}]' if unit else name, fontsize=9)
        ax.set_title(name, color='#58A6FF', fontsize=10, pad=4)
        ax.grid(color='#21262D', lw=0.5)
        ax.legend(fontsize=7, facecolor='#1C2128', labelcolor='white',
                  edgecolor='#30363D', loc='best')

    # Table of exit conditions
    ax_tbl = axes.flat[5]
    ax_tbl.set_facecolor('#161B22')
    ax_tbl.axis('off')
    rows_data = []
    for sol, lbl in zip(solutions, labels):
        rows_data.append([
            lbl[:30],
            f"{sol['M_exit']:.4f}",
            f"{sol['P_exit']/1e3:.2f}",
            f"{sol['T_exit']:.1f}",
            f"{sol['rho'][-1]:.5f}",
            f"{sol['V'][-1]:.1f}",
        ])
    tbl = ax_tbl.table(
        cellText=rows_data,
        colLabels=['Case','M_exit','P_exit\n(kPa)','T_exit\n(K)','ρ_exit\n(kg/m³)','V_exit\n(m/s)'],
        cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1C2128' if r > 0 else '#21262D')
        cell.set_text_props(color='#C9D1D9' if r > 0 else '#58A6FF',
                            fontfamily='monospace')
        cell.set_edgecolor('#30363D')
    ax_tbl.set_title('Exit Conditions Summary', color='#58A6FF', fontsize=10, pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buf1 = io.BytesIO()
    fig.savefig(buf1, format='png', dpi=100, bbox_inches='tight', facecolor='#0D1117')
    buf1.seek(0)
    img1_b64 = base64.b64encode(buf1.read()).decode('utf-8')

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight', facecolor='#0D1117')
    buf2.seek(0)
    img2_b64 = base64.b64encode(buf2.read()).decode('utf-8')

    plt.close('all')
    return [img1_b64, img2_b64]


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENT TASKS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌─────────────────────────────────────────────────────────────────────────┐
│  LAB NOTEBOOK — Experiment 4                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  TASK 1: At the exit (design condition), manually verify:               │
│    T_exit = T₀ / (1 + (γ-1)/2 · M²)                                   │
│    P_exit = P₀ / (1 + (γ-1)/2 · M²)^(γ/(γ-1))                        │
│    Compare with printed values. Discrepancy < 0.01%?                   │
│                                                                         │
│  TASK 2: Compute the thrust force using the exit momentum:              │
│    F = ṁ · V_exit + (P_exit - P_atm) · A_exit                         │
│    Use P_atm = 101,325 Pa                                               │
│                                                                         │
│  TASK 3: Compare the 2-D color maps for M, P, T.                       │
│    Which property has the SHARPEST gradient at the throat?             │
│    Explain why physically.                                              │
│                                                                         │
│  TASK 4: For the shock case, what happens to T across the shock?       │
│    Does it increase or decrease? Use the T2/T1 formula to predict.     │
│                                                                         │
│  TASK 5: Why does velocity DECREASE after a normal shock               │
│    even though temperature INCREASES?                                  │
└─────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == '__main__':
    run_flow_field()
