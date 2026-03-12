"""
app.py — Flask web server for C-D Nozzle Virtual Lab
Run:  python app.py
Open: http://127.0.0.1:5000
"""

# Fix Windows console encoding for Unicode box-drawing characters
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── matplotlib MUST use non-interactive backend BEFORE any pyplot import ──────
# Redirect font/config cache to /tmp for read-only serverless environments (Vercel)
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp')

import matplotlib
matplotlib.use('Agg')

import traceback
import concurrent.futures

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify

# ── Nozzle physics (for fast stats computation) ───────────────────────────────
from nozzle_physics import (
    NozzleFlowSolver, mach_from_area_ratio, area_mach_ratio,
    isentropic_P_ratio, isentropic_T_ratio,
    normal_shock_M2, normal_shock_P_ratio,
    normal_shock_T_ratio, normal_shock_P0_ratio,
    sonic_velocity, R_AIR, GAMMA_AIR,
)

# ── Experiment plot functions ─────────────────────────────────────────────────
from experiment_01_mach_distribution         import plot_mach_distribution
from experiment_02_normal_shock              import shock_sweep
from experiment_03_throat_sonic_choking      import run_choking_experiment
from experiment_04_full_flow_field           import run_flow_field
from experiment_05_nozzle_design_optimization import run_optimisation, mass_flow

# Use absolute path so Flask finds templates in all environments (including Vercel)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_BASE_DIR, 'templates'))

TIMEOUT_S = 90   # seconds before a simulation is killed

# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default

def _i(d, key, default=80):
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        return default

def _csv(headers, rows):
    """Build a CSV string: headers is list of strings, rows is list of dicts."""
    lines = [','.join(headers)]
    for row in rows:
        lines.append(','.join(str(row.get(h, '')) for h in headers))
    return '\n'.join(lines)

def _run_timeout(fn, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        return future.result(timeout=TIMEOUT_S)

def _short_tb(tb_str):
    """Return last 6 lines of a traceback for the UI summary."""
    lines = tb_str.strip().splitlines()
    return '\n'.join(lines[-6:])

# ── Stats + CSV computation (no matplotlib, very fast) ─────────────────────────

def _stats1(d):
    G = GAMMA_AIR
    P0 = _f(d,'P0',200000); T0 = _f(d,'T0',300); AR = _f(d,'AR_exit',3.0)
    solver = NozzleFlowSolver(P0=P0, T0=T0, AR_exit=AR, n_pts=250)
    sol = solver.solve(solver.Pb_supersonic)
    rows = [{'x':f'{x:.5f}','M':f'{M:.5f}','P_Pa':f'{P:.3f}',
              'T_K':f'{T:.3f}','rho_kgm3':f'{r:.7f}','V_ms':f'{V:.3f}'}
            for x,M,P,T,r,V in zip(sol['x'],sol['M'],sol['P'],sol['T'],sol['rho'],sol['V'])]
    csv = _csv(['x','M','P_Pa','T_K','rho_kgm3','V_ms'], rows)
    return {
        'M_exit': round(sol['M_exit'], 4),
        'P_exit_kPa': round(sol['P_exit']/1e3, 3),
        'T_exit_K': round(sol['T_exit'], 2),
        'V_exit_ms': round(sol['V'][-1], 2),
        'Pb_design_kPa': round(solver.Pb_supersonic/1e3, 3),
        'Pb_subsonic_kPa': round(solver.Pb_subsonic/1e3, 2),
        'Pb_shock_exit_kPa': round(solver.Pb_shock_at_exit/1e3, 2),
    }, csv

def _stats2(d):
    P0 = _f(d,'P0',300000); T0 = _f(d,'T0',350); AR = _f(d,'AR_exit',3.5)
    solver = NozzleFlowSolver(P0=P0, T0=T0, AR_exit=AR, n_pts=250)
    Pb_mid = (solver.Pb_shock_at_exit + solver.Pb_subsonic) * 0.5
    sol = solver.solve(Pb_mid)
    shock_d = {}
    if sol['shock_idx'] is not None:
        M1 = sol['M'][sol['shock_idx']-1]
        shock_d = {
            'M1': round(M1, 4), 'M2': round(normal_shock_M2(M1), 4),
            'P2_P1': round(normal_shock_P_ratio(M1), 4),
            'T2_T1': round(normal_shock_T_ratio(M1), 4),
            'P02_P01': round(normal_shock_P0_ratio(M1), 4),
            'ds_R': round(-np.log(normal_shock_P0_ratio(M1)), 4),
        }
    rows = [{'x':f'{x:.5f}','M':f'{M:.5f}','P_Pa':f'{P:.3f}'}
            for x,M,P in zip(sol['x'],sol['M'],sol['P'])]
    csv = _csv(['x','M','P_Pa'], rows)
    return {
        'M_design': round(solver.M_exit_design, 4),
        'Pb_design_kPa': round(solver.Pb_supersonic/1e3, 3),
        'Pb_shock_exit_kPa': round(solver.Pb_shock_at_exit/1e3, 3),
        'Pb_subsonic_kPa': round(solver.Pb_subsonic/1e3, 2),
        **shock_d,
    }, csv

def _stats3(d):
    G = GAMMA_AIR
    P0 = _f(d,'P0',200000); T0 = _f(d,'T0',300); A = _f(d,'A_THROAT',0.01)
    Pcrit = (2/(G+1))**(G/(G-1))
    mdot_max = A * P0 / T0**0.5 * (G/R_AIR)**0.5 * (2/(G+1))**((G+1)/(2*(G-1)))
    T_star = T0 * 2/(G+1)
    P_star = P0 * Pcrit
    rho_star = P_star / (R_AIR * T_star)
    a_star = (G * R_AIR * T_star)**0.5
    Pb_arr = np.linspace(0.05, 1.0, 60)
    rows = []
    for ratio in Pb_arr:
        if ratio <= Pcrit:
            md = mdot_max * 1000
        else:
            bv = ratio**((G-1)/G)
            Mt = min((2/(G-1)*(1/bv-1))**0.5, 1.0)
            Tt = T0/(1+(G-1)/2*Mt**2)
            rt = ratio*P0/(R_AIR*Tt)
            md = rt * Mt * (G*R_AIR*Tt)**0.5 * A * 1000
        rows.append({'Pb_over_P0':f'{ratio:.5f}','mdot_gs':f'{md:.5f}'})
    csv = _csv(['Pb_over_P0','mdot_gs'], rows)
    return {
        'mdot_max_gs': round(mdot_max*1000, 4),
        'Pcrit': round(Pcrit, 4),
        'Pcrit_kPa': round(P_star/1e3, 3),
        'T_star_K': round(T_star, 2),
        'P_star_kPa': round(P_star/1e3, 3),
        'rho_star': round(rho_star, 5),
        'a_star_ms': round(a_star, 2),
    }, csv

def _stats4(d):
    G = GAMMA_AIR
    P0 = _f(d,'P0',500000); T0 = _f(d,'T0',500)
    AR = _f(d,'AR_exit',4.0); A = _f(d,'A_THROAT',0.005)
    solver = NozzleFlowSolver(P0=P0, T0=T0, AR_exit=AR, n_pts=250)
    sol = solver.solve(solver.Pb_supersonic)
    V_e = sol['V'][-1]; rho_e = sol['rho'][-1]
    thrust = rho_e * V_e**2 * A * AR
    rows = [{'x':f'{x:.5f}','M':f'{M:.5f}','P_Pa':f'{P:.3f}',
              'T_K':f'{T:.3f}','rho_kgm3':f'{r:.7f}','V_ms':f'{V:.3f}'}
            for x,M,P,T,r,V in zip(sol['x'],sol['M'],sol['P'],sol['T'],sol['rho'],sol['V'])]
    csv = _csv(['x','M','P_Pa','T_K','rho_kgm3','V_ms'], rows)
    return {
        'M_exit': round(sol['M_exit'], 4),
        'P_exit_kPa': round(sol['P_exit']/1e3, 3),
        'T_exit_K': round(sol['T_exit'], 2),
        'V_exit_ms': round(V_e, 2),
        'rho_exit': round(rho_e, 6),
        'thrust_N': round(thrust, 2),
    }, csv

def _stats5(d):
    G = GAMMA_AIR
    P0 = _f(d,'P0',1e6); T0 = _f(d,'T0',2800); A_star = _f(d,'A_STAR',0.01)
    AR_MAX = _f(d,'AR_MAX',30.0)
    alts = {
        'Sea Level': _f(d,'alt_sea',101325),
        '10 km':     _f(d,'alt_10km',26500),
        '30 km':     _f(d,'alt_30km',1197),
        'Space':     _f(d,'alt_space',1.0),
    }
    mdot = mass_flow(P0, T0, A_star, G)
    g0 = 9.81
    alt_rows = []
    for name, Patm in alts.items():
        if Patm >= P0:
            AR_opt = 1.0; M_e = 1.0
        else:
            M_e = ((2/(G-1))*((P0/Patm)**((G-1)/G)-1))**0.5
            AR_opt = area_mach_ratio(M_e, G)
        M_e2 = mach_from_area_ratio(AR_opt, supersonic=True) if AR_opt > 1.0 else 1.0
        T_e = T0 / (1+(G-1)/2*M_e2**2)
        Ve = M_e2 * (G*R_AIR*T_e)**0.5
        Pe = P0 / isentropic_P_ratio(M_e2, G)
        F  = mdot * Ve + (Pe - Patm) * A_star * AR_opt
        Cf = F / (P0 * A_star)
        Isp = F / (mdot * g0)
        alt_rows.append({'alt':name,'AR_opt':round(AR_opt,2),'M_exit':round(M_e2,3),
                         'Cf':round(Cf,4),'F_N':round(F,1),'Isp_s':round(Isp,1)})
    AR_arr = np.linspace(1.05, AR_MAX, 150)
    Patm0 = _f(d,'alt_sea',101325)
    csv_rows = []
    for AR in AR_arr:
        Me = mach_from_area_ratio(AR, supersonic=True)
        Te = T0/(1+(G-1)/2*Me**2)
        Ve = Me*(G*R_AIR*Te)**0.5
        Pe = P0/isentropic_P_ratio(Me,G)
        F  = mdot*Ve+(Pe-Patm0)*A_star*AR
        csv_rows.append({'AR':f'{AR:.4f}','V_ms':f'{Ve:.2f}','Cf_sealevel':f'{F/(P0*A_star):.5f}'})
    csv = _csv(['AR','V_ms','Cf_sealevel'], csv_rows)
    return {'mdot_gs': round(mdot*1000, 4), 'alts': alt_rows}, csv

STATS_FN = {1: _stats1, 2: _stats2, 3: _stats3, 4: _stats4, 5: _stats5}

# ── Route helpers ─────────────────────────────────────────────────────────────

def _respond(n, plot_fn, plot_kwargs):
    d = request.get_json(force=True)
    try:
        images = _run_timeout(plot_fn, **plot_kwargs(d))
        stats, csv = STATS_FN[n](d)
        return jsonify({'success': True, 'images': images, 'stats': stats, 'csv': csv})
    except concurrent.futures.TimeoutError:
        return jsonify({'success': False,
                        'error': f'Simulation timed out after {TIMEOUT_S} s.\n'
                                  'Try smaller N_PB_POINTS or AR_MAX.'})
    except Exception:
        tb = traceback.format_exc()
        return jsonify({'success': False,
                        'error': _short_tb(tb),
                        'full_error': tb})

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run/1', methods=['POST'])
def run_exp1():
    return _respond(1, plot_mach_distribution, lambda d: dict(
        P0=_f(d,'P0',200000), T0=_f(d,'T0',300), AR_exit=_f(d,'AR_exit',3.0),
        pb_ratio_subsonic=_f(d,'pb_ratio_subsonic',0.95),
        pb_ratio_shock_near=_f(d,'pb_ratio_shock_near',0.40),
        pb_ratio_shock_mid=_f(d,'pb_ratio_shock_mid',0.55),
    ))

@app.route('/run/2', methods=['POST'])
def run_exp2():
    def kw(d):
        raw = d.get('custom_pb_list','').strip()
        pb  = [float(x.strip())*1e3 for x in raw.split(',') if x.strip()] if raw else None
        return dict(P0=_f(d,'P0',300000), T0=_f(d,'T0',350),
                    AR_exit=_f(d,'AR_exit',3.5), CUSTOM_Pb_LIST=pb)
    return _respond(2, shock_sweep, kw)

@app.route('/run/3', methods=['POST'])
def run_exp3():
    return _respond(3, run_choking_experiment, lambda d: dict(
        P0=_f(d,'P0',200000), T0=_f(d,'T0',300), AR_exit=_f(d,'AR_exit',3.0),
        A_THROAT=_f(d,'A_THROAT',0.01), N_PB_POINTS=_i(d,'N_PB_POINTS',80),
    ))

@app.route('/run/4', methods=['POST'])
def run_exp4():
    def kw(d):
        c = d.get('COMPARISON','shock_mid')
        return dict(P0=_f(d,'P0',500000), T0=_f(d,'T0',500),
                    AR_exit=_f(d,'AR_exit',4.0), A_THROAT=_f(d,'A_THROAT',0.005),
                    COMPARISON=None if c=='none' else c)
    return _respond(4, run_flow_field, kw)

@app.route('/run/5', methods=['POST'])
def run_exp5():
    def kw(d):
        return dict(
            P0=_f(d,'P0',1e6), T0=_f(d,'T0',2800),
            A_STAR=_f(d,'A_STAR',0.01), AR_MAX=_f(d,'AR_MAX',30.0),
            ALTITUDES={
                'Sea Level (0 km)' : _f(d,'alt_sea',101325),
                '10 km (cruise)'   : _f(d,'alt_10km',26500),
                '30 km (stratosph)': _f(d,'alt_30km',1197),
                'Space (vacuum)'   : _f(d,'alt_space',1.0),
            })
    return _respond(5, run_optimisation, kw)

# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*55)
    print('  C-D Nozzle Virtual Lab — Web Server')
    print('  Open browser at:  http://127.0.0.1:5000')
    print('='*55 + '\n')
    app.run(debug=False, port=5000, threaded=False)
