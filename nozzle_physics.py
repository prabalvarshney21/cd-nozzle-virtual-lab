"""
nozzle_physics.py
=================
Core isentropic and normal-shock relations for a Convergent-Divergent (De Laval) Nozzle.
All equations referenced to Anderson, "Modern Compressible Flow", 3rd Ed.

Students should NOT modify this file — it is the physics engine.
"""

import numpy as np


def _brentq(f, a, b, xtol=1e-12, maxiter=500):
    """Brent's method root-finder (replaces scipy.optimize.brentq)."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(maxiter):
        c = (a + b) / 2.0
        fc = f(c)
        if abs(b - a) < xtol or fc == 0:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2.0


# ── Constants ──────────────────────────────────────────────────────────────────
GAMMA_AIR = 1.4          # Ratio of specific heats, air
R_AIR     = 287.0        # Specific gas constant, J/(kg·K)
CP_AIR    = R_AIR * GAMMA_AIR / (GAMMA_AIR - 1)   # J/(kg·K)


# ══════════════════════════════════════════════════════════════════════════════
# ISENTROPIC RELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def isentropic_T_ratio(M, gamma=GAMMA_AIR):
    """T0/T = 1 + (gamma-1)/2 * M^2"""
    return 1.0 + (gamma - 1) / 2.0 * M**2

def isentropic_P_ratio(M, gamma=GAMMA_AIR):
    """P0/P = (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1))"""
    return isentropic_T_ratio(M, gamma) ** (gamma / (gamma - 1))

def isentropic_rho_ratio(M, gamma=GAMMA_AIR):
    """rho0/rho = (1 + (gamma-1)/2 * M^2)^(1/(gamma-1))"""
    return isentropic_T_ratio(M, gamma) ** (1.0 / (gamma - 1))

def area_mach_ratio(M, gamma=GAMMA_AIR):
    """
    A/A* = (1/M) * [(2/(gamma+1)) * (1 + (gamma-1)/2 * M^2)]^((gamma+1)/(2*(gamma-1)))
    Valid for M > 0.
    """
    if np.isscalar(M):
        if M == 0:
            return np.inf
    term = (2 / (gamma + 1)) * isentropic_T_ratio(M, gamma)
    exp  = (gamma + 1) / (2 * (gamma - 1))
    return (1.0 / M) * term**exp

def mach_from_area_ratio(AR, supersonic=True, gamma=GAMMA_AIR):
    """
    Invert A/A* = f(M) numerically.
    AR   : A/A* value (>=1)
    supersonic : if True, return the M>1 solution; else M<1
    """
    if AR < 1.0:
        raise ValueError(f"A/A* must be >= 1, got {AR:.4f}")
    if AR == 1.0:
        return 1.0

    if supersonic:
        M = _brentq(lambda m: area_mach_ratio(m, gamma) - AR, 1.0 + 1e-9, 50.0)
    else:
        M = _brentq(lambda m: area_mach_ratio(m, gamma) - AR, 1e-9, 1.0 - 1e-9)
    return M

def sonic_velocity(T, gamma=GAMMA_AIR, R=R_AIR):
    """a = sqrt(gamma * R * T)  [m/s]"""
    return np.sqrt(gamma * R * T)


# ══════════════════════════════════════════════════════════════════════════════
# NORMAL SHOCK RELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def normal_shock_M2(M1, gamma=GAMMA_AIR):
    """M2 downstream of normal shock"""
    num = (gamma - 1) * M1**2 + 2
    den = 2 * gamma * M1**2 - (gamma - 1)
    return np.sqrt(num / den)

def normal_shock_P_ratio(M1, gamma=GAMMA_AIR):
    """P2/P1 across normal shock"""
    return 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)

def normal_shock_T_ratio(M1, gamma=GAMMA_AIR):
    """T2/T1 across normal shock"""
    return normal_shock_P_ratio(M1, gamma) * (2 + (gamma - 1) * M1**2) / ((gamma + 1) * M1**2)

def normal_shock_rho_ratio(M1, gamma=GAMMA_AIR):
    """rho2/rho1 = (gamma+1)*M1^2 / (2 + (gamma-1)*M1^2)"""
    return (gamma + 1) * M1**2 / (2 + (gamma - 1) * M1**2)

def normal_shock_P0_ratio(M1, gamma=GAMMA_AIR):
    """P02/P01 — stagnation pressure ratio (entropy increase indicator)"""
    M2 = normal_shock_M2(M1, gamma)
    return (isentropic_P_ratio(M2, gamma) / isentropic_P_ratio(M1, gamma))


# ══════════════════════════════════════════════════════════════════════════════
# NOZZLE GEOMETRY  (parabolic contour, standard design)
# ══════════════════════════════════════════════════════════════════════════════

def nozzle_geometry(n_points=500, AR_exit=3.0, AR_entry=4.0, throat_x=0.45):
    """
    Generate non-dimensional area distribution A(x)/A*
    x in [0, 1]
    throat at x = throat_x

    Returns:
        x    : array of x positions
        AR   : array of A/A* values
    """
    x = np.linspace(0, 1, n_points)
    AR = np.zeros(n_points)

    for i, xi in enumerate(x):
        if xi <= throat_x:
            # Converging: smooth parabola from AR_entry → 1
            t = xi / throat_x           # 0→1
            AR[i] = AR_entry - (AR_entry - 1) * t**1.8
        else:
            # Diverging: parabola from 1 → AR_exit
            t = (xi - throat_x) / (1 - throat_x)  # 0→1
            AR[i] = 1.0 + (AR_exit - 1) * t**1.8

    # Throat exactly = 1
    throat_idx = np.argmin(np.abs(x - throat_x))
    AR[throat_idx] = 1.0
    return x, AR


# ══════════════════════════════════════════════════════════════════════════════
# FLOW-FIELD SOLVER  — determine operating condition given back pressure
# ══════════════════════════════════════════════════════════════════════════════

class NozzleFlowSolver:
    """
    Solves the 1-D quasi-steady flow through a C-D nozzle.

    Parameters
    ----------
    P0      : stagnation (reservoir) pressure, Pa
    T0      : stagnation temperature, K
    AR_exit : exit-to-throat area ratio A_e/A*
    AR_entry: entry-to-throat area ratio A_in/A*
    n_pts   : number of spatial points
    gamma   : ratio of specific heats
    """

    def __init__(self, P0=200_000, T0=300.0, AR_exit=3.0,
                 AR_entry=4.0, n_pts=500, gamma=GAMMA_AIR):
        self.P0       = P0
        self.T0       = T0
        self.AR_exit  = AR_exit
        self.AR_entry = AR_entry
        self.n_pts    = n_pts
        self.gamma    = gamma
        self.g        = gamma

        self.x, self.AR = nozzle_geometry(n_pts, AR_exit, AR_entry)
        self.throat_idx  = np.argmin(self.AR)

        # Precompute critical back-pressures
        self._compute_critical_pressures()

    # ── Critical pressure ratios ──────────────────────────────────────────────

    def _compute_critical_pressures(self):
        """Compute the four critical Pb/P0 values that separate flow regimes."""
        g = self.g
        AR_e = self.AR_exit

        # 1) Fully subsonic: Mach at exit (subsonic root)
        M_sub_e = mach_from_area_ratio(AR_e, supersonic=False, gamma=g)
        self.Pb_subsonic = self.P0 / isentropic_P_ratio(M_sub_e, g)

        # 2) Fully isentropic supersonic
        M_sup_e = mach_from_area_ratio(AR_e, supersonic=True, gamma=g)
        self.Pb_supersonic = self.P0 / isentropic_P_ratio(M_sup_e, g)
        self.M_exit_design = M_sup_e

        # 3) Normal shock AT exit
        M2_after_shock = normal_shock_M2(M_sup_e, g)
        P2_over_P1     = normal_shock_P_ratio(M_sup_e, g)
        # P0_behind_shock = P0 * P02/P01
        P0_behind = self.P0 * normal_shock_P0_ratio(M_sup_e, g)
        self.Pb_shock_at_exit = P0_behind / isentropic_P_ratio(M2_after_shock, g)

        # 4) Choking begins: P_back such that throat just goes sonic
        #    (Pb equals subsonic solution for this geometry)
        self.Pb_choked = self.Pb_subsonic   # same as subsonic-isentropic for given AR

    # ── Public solve method ───────────────────────────────────────────────────

    def solve(self, Pb):
        """
        Given back pressure Pb (Pa), return arrays of flow properties.

        Returns dict with keys:
          x, AR, M, P, T, rho, V, a  (all length n_pts)
          regime  : string description
          shock_x : x-location of shock (or None)
          M_exit, P_exit, T_exit
        """
        Pb_ratio = Pb / self.P0
        g        = self.g

        # Determine regime
        if Pb >= self.Pb_subsonic:
            return self._solve_subsonic_unchoked(Pb)
        elif Pb >= self.Pb_shock_at_exit:
            return self._solve_shock_in_nozzle(Pb)
        elif Pb > self.Pb_supersonic:
            return self._solve_overexpanded(Pb)
        else:
            return self._solve_isentropic_supersonic()

    # ── Regime solvers ────────────────────────────────────────────────────────

    def _isentropic_properties(self, M_arr):
        """Vectorised isentropic relations."""
        g  = self.g
        T  = self.T0 / isentropic_T_ratio(M_arr, g)
        P  = self.P0 / isentropic_P_ratio(M_arr, g)
        rho= P / (R_AIR * T)
        a  = sonic_velocity(T, g)
        V  = M_arr * a
        return T, P, rho, V, a

    def _subsonic_mach_profile(self):
        """Mach array for fully subsonic (unchoked) flow."""
        M = np.array([mach_from_area_ratio(ar, supersonic=False, gamma=self.g)
                      for ar in self.AR])
        return M

    def _supersonic_mach_profile(self):
        """Mach array for isentropic supersonic diverging section."""
        M = np.zeros(self.n_pts)
        ti = self.throat_idx
        # Converging: subsonic
        for i in range(ti + 1):
            M[i] = mach_from_area_ratio(self.AR[i], supersonic=False, gamma=self.g)
        M[ti] = 1.0
        # Diverging: supersonic
        for i in range(ti + 1, self.n_pts):
            M[i] = mach_from_area_ratio(self.AR[i], supersonic=True, gamma=self.g)
        return M

    def _solve_subsonic_unchoked(self, Pb):
        M = self._subsonic_mach_profile()
        T, P, rho, V, a = self._isentropic_properties(M)
        return dict(x=self.x, AR=self.AR, M=M, P=P, T=T, rho=rho, V=V, a=a,
                    regime="Subsonic (unchoked)",
                    shock_x=None, shock_idx=None,
                    M_exit=M[-1], P_exit=P[-1], T_exit=T[-1])

    def _solve_isentropic_supersonic(self):
        M = self._supersonic_mach_profile()
        T, P, rho, V, a = self._isentropic_properties(M)
        return dict(x=self.x, AR=self.AR, M=M, P=P, T=T, rho=rho, V=V, a=a,
                    regime="Fully Isentropic Supersonic (design condition)",
                    shock_x=None, shock_idx=None,
                    M_exit=M[-1], P_exit=P[-1], T_exit=T[-1])

    def _solve_shock_in_nozzle(self, Pb):
        """Locate normal shock position in diverging section."""
        g   = self.g
        M_sup = self._supersonic_mach_profile()
        ti  = self.throat_idx

        def residual(shock_idx_f):
            si = int(shock_idx_f)
            si = max(ti + 1, min(si, self.n_pts - 2))
            M1 = M_sup[si]
            # Post-shock: subsonic, find M2
            M2       = normal_shock_M2(M1, g)
            P0_ratio = normal_shock_P0_ratio(M1, g)
            P0_new   = self.P0 * P0_ratio
            # Solve rest of nozzle subsonically with new P0
            AR_post  = self.AR[si:]
            M_sub    = np.array([mach_from_area_ratio(ar * (self.AR[si] / self.AR[si]),
                                                       supersonic=False, gamma=g)
                                  for ar in AR_post])
            # Correct: reference throat is virtual (A*/after shock > original A*)
            AR_virtual = AR_post / AR_post[0] * area_mach_ratio(M2, g)
            M_sub = np.array([mach_from_area_ratio(ar, supersonic=False, gamma=g)
                               for ar in AR_virtual])
            P_exit_after = P0_new / isentropic_P_ratio(M_sub[-1], g)
            return P_exit_after - Pb

        # Binary search for shock location
        lo, hi = float(ti + 1), float(self.n_pts - 2)
        try:
            shock_idx = int(_brentq(residual, lo, hi, xtol=1.0))
        except ValueError:
            shock_idx = self.n_pts - 2

        # Build complete M-array
        M = np.zeros(self.n_pts)
        M[:shock_idx + 1] = M_sup[:shock_idx + 1]

        M1_shock = M_sup[shock_idx]
        M2_shock = normal_shock_M2(M1_shock, g)
        P0_ratio = normal_shock_P0_ratio(M1_shock, g)
        P0_new   = self.P0 * P0_ratio

        AR_post     = self.AR[shock_idx:]
        AR_virtual  = AR_post / AR_post[0] * area_mach_ratio(M2_shock, g)
        M_post = np.array([mach_from_area_ratio(ar, supersonic=False, gamma=g)
                            for ar in AR_virtual])
        M[shock_idx:] = M_post

        # Properties (two-zone P0)
        T  = np.zeros(self.n_pts)
        P  = np.zeros(self.n_pts)
        rho= np.zeros(self.n_pts)
        V  = np.zeros(self.n_pts)
        a  = np.zeros(self.n_pts)

        # Pre-shock: original P0, T0
        T[:shock_idx], P[:shock_idx], rho[:shock_idx], V[:shock_idx], a[:shock_idx] = \
            self._isentropic_properties(M[:shock_idx])
        # Shock jump
        P_jump = P[shock_idx - 1] * normal_shock_P_ratio(M1_shock, g)
        T_jump = T[shock_idx - 1] * normal_shock_T_ratio(M1_shock, g)
        # Post-shock: new P0, T0 (T0 unchanged across shock)
        T[shock_idx:] = self.T0 / isentropic_T_ratio(M_post, g)
        P[shock_idx:] = P0_new / isentropic_P_ratio(M_post, g)
        rho[shock_idx:] = P[shock_idx:] / (R_AIR * T[shock_idx:])
        a[shock_idx:] = sonic_velocity(T[shock_idx:], g)
        V[shock_idx:] = M[shock_idx:] * a[shock_idx:]

        return dict(x=self.x, AR=self.AR, M=M, P=P, T=T, rho=rho, V=V, a=a,
                    regime="Normal Shock in Diverging Section",
                    shock_x=self.x[shock_idx], shock_idx=shock_idx,
                    M_exit=M[-1], P_exit=P[-1], T_exit=T[-1])

    def _solve_overexpanded(self, Pb):
        """Supersonic with oblique/expansion outside nozzle (modelled as isentropic inside)."""
        result = self._solve_isentropic_supersonic()
        result['regime'] = f"Over-expanded (external oblique shock, Pb/P_exit={Pb/result['P_exit']:.3f})"
        return result
