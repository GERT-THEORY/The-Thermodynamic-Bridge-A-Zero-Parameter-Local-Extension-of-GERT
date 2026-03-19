"""
GERT LOCAL MINI-FRAMEWORK — Internal Test v0.1
================================================
Laboratory notebook — not for publication yet.

Two objectives:
  1. Test whether the GERT local extension produces physically reasonable
     rotation curves using ONLY functions and parameters from Paper 1.
  2. Verify that the extension does NOT falsify the existing GERT framework:
     - Solar System correction must be zero (or negligible)
     - Cosmological parameters must be untouched
     - No new free parameters introduced beyond α (the retention fraction)

Physical hypothesis:
  Bound structures are cohesive relics of the Constructive Era, embedded in
  a globally entropic background. The entropic Work that is "trapped" inside
  the gravitational potential well of a bound system manifests as an effective
  additional inward acceleration, proportional to how much the local entropic
  mode is activated AND how much cohesion has already declined from its
  builder-era maximum.

Equation-pilot:
  g_GERT(r) = g_bar(r) * [1 + α · f_L(x_loc(r)) · (1 - f_M(x_loc(r))/f_M_i)]

where:
  x_loc(r) = log10( 3 M_b(<r) / (4π r³) )   [local mean baryonic density]
  f_M, f_L = EXACT functions from Paper 1    [zero new parameters]
  α         = single free parameter          [entropic retention fraction]

Authors: Veronica + Claude (internal test)
Date:    March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit   # logistic σ(x) = 1/(1+e^-x)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 0 — PAPER 1 PARAMETERS (exact, frozen, no modification allowed)
# ─────────────────────────────────────────────────────────────────────────────

# Cohesive sector (fM)
FM_I       =  0.7831    # initial matter factor (builder era)
FM_F       =  0.5851    # final matter factor   (maintainer era)
LOG_RHO_M  = -20.30     # matter transition centre [log10 kg/m³]
D_M        =  1.0       # matter transition width  [dex]
FM_PEAK    =  0.37      # cohesive peak amplitude
LOG_RHO_C  = -17.41     # cohesive peak centre     [log10 kg/m³]
SIGMA_C    =  1.0       # cohesive peak width       [dex]

# Entropic sector (fL)
FL_I       =  1.3414    # initial entropic factor
FL_M       =  1.1236    # mid/final entropic factor (asymptotic base)
LOG_RHO_L  = -25.60     # entropic transition centre [log10 kg/m³]
D_L        =  2.0       # entropic transition width  [dex]
FL_PEAK    =  4.6245    # entropic peak amplitude
LOG_RHO_L2 = -23.93     # entropic peak centre       [log10 kg/m³]
SIGMA_L2   =  1.0       # entropic peak width        [dex]
K_GAS      =  0.143     # gas term amplitude
X_GAS      = -26.750    # gas activation density     [log10 kg/m³]
GAMMA_GAS  =  0.50      # gas slope

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — EXACT PAPER 1 FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def logistic(x, x0, delta):
    """GERT logistic: σ((x - x0)/delta)  →  0 at x>>x0, 1 at x<<x0"""
    return expit(-(x - x0) / delta)

def gaussian(x, x0, sigma):
    """GERT Gaussian peak: exp(-0.5*((x-x0)/sigma)^2)"""
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def fM(x):
    """
    Cohesive factor — matching original MCMC script, corrected for logistic direction.

    Original script: logist → 1 at HIGH density, → 0 at LOW density
    My logistic:     expit  → 0 at HIGH density, → 1 at LOW density

    Equivalent formula with my logistic direction:
      base = fM_i + (fM_f - fM_i) * logistic(x, logρ_M, dM)
        → fM_i at high density (logistic→0)  ← builder era value ✓
        → fM_f at low density  (logistic→1)  ← maintainer era value ✓

    Peak is MULTIPLICATIVE on base (matching eM_unified in original script):
      correction = base * fM_peak * G(x; logρ_c, σ_c)
    This means fM can exceed fM_i near log ρ_c = -17.41 (recombination boost).
    Screening factor S(x) floors at zero to handle this (Option A).
    """
    base = FM_I + (FM_F - FM_I) * logistic(x, LOG_RHO_M, D_M)
    correction = base * FM_PEAK * gaussian(x, LOG_RHO_C, SIGMA_C)
    return base + correction

def fL(x):
    """
    Entropic factor — matching original MCMC script, corrected for logistic direction.

    Equivalent formula with my logistic direction:
      base = fL_i + (fL_m - fL_i) * logistic(x, logρ_L, dL)
        → fL_i at high density (logistic→0)  ← early universe ✓
        → fL_m at low density  (logistic→1)  ← asymptotic base ✓

    Gas term: active below x_gas (same as original)
    Peak is MULTIPLICATIVE over (base + gas):
      correction = (base + gas) * fL_peak * G(x; logρ_L2, σ_L2)
    """
    base  = FL_I + (FL_M - FL_I) * logistic(x, LOG_RHO_L, D_L)
    gas   = K_GAS * np.maximum(0.0, np.exp((X_GAS - x) / GAMMA_GAS) - 1.0)
    total = base + gas
    correction = total * FL_PEAK * gaussian(x, LOG_RHO_L2, SIGMA_L2)
    return total + correction




# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — LOCAL EXTENSION
# ─────────────────────────────────────────────────────────────────────────────

def x_local(r_m, Mb_enclosed_kg):
    """
    Local thermodynamic state variable.
    x_loc(r) = log10( 3 M_b(<r) / (4π r³) )   [kg/m³ → log10]
    Exact analogue of the cosmological x = log10(ρ_m).
    """
    rho_local = 3.0 * Mb_enclosed_kg / (4.0 * np.pi * r_m**3)
    return np.log10(np.maximum(rho_local, 1e-40))

def screening_factor(x):
    """
    Cohesive screening: how much has cohesion declined from its builder maximum?

    S(x) = max(0,  1 - fM(x) / fM_i)     ← OPTION A: floor at zero

    Physical logic (Option A):
    - fM(x) < fM_i  (cohesion has declined)  → S > 0 → correction acts
    - fM(x) = fM_i  (Solar System, high density) → S = 0 → correction = 0 exactly
    - fM(x) > fM_i  (recombination boost peak)   → S = 0 → correction = 0
      Interpretation: regions denser than the builder-era maximum are in FULL
      constructive dominance — no entropic Work retention needed. Physically
      this covers molecular clouds, stellar interiors, compact objects.

    This means the GERT local correction is ONLY active where cohesion has
    already declined from its initial value — precisely where dark matter
    effects are observationally most prominent.
    """
    return np.maximum(0.0,  1.0 - fM(x) / FM_I)

def g_GERT(g_bar, x, alpha):
    """
    GERT local effective acceleration.
    g_GERT = g_bar * [1 + α · fL(x) · S(x)]

    Physical reading:
      fL(x)  = how much entropic Work is activated at local density x
      S(x)   = how much of that Work is NOT screened by cohesion
      α      = fraction of retained entropic Work converted to effective gravity
    """
    correction = alpha * fL(x) * screening_factor(x)
    return g_bar * (1.0 + correction)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — DIAGNOSTIC: fM and fL across galactic density range
# ─────────────────────────────────────────────────────────────────────────────

def plot_functions_local_range():
    """
    Show fM, fL, and the correction factor across galactic density scales.
    Red lines mark the cosmological GERT milestones.
    """
    x = np.linspace(-28, -14, 1000)

    fm = fM(x)
    fl = fL(x)
    screen = screening_factor(x)
    # total multiplicative correction at α=0.5
    corr  = 0.5 * fl * screen

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GERT Local — Paper 1 Functions in Galactic Density Range",
                 fontsize=14, fontweight='bold')

    milestones = {
        r'$\log\rho_c=-17.41$\n(cohesive peak)':   -17.41,
        r'$\log\rho_M=-20.30$\n(builder→maint.)':  -20.30,
        r'$\log\rho_{L2}=-23.93$\n(entropic peak)': -23.93,
        r'$\log\rho_L=-25.60$\n(entr. transition)': -25.60,
    }
    colors_ms = ['#d62728','#ff7f0e','#2ca02c','#1f77b4']

    # Panel 1: fM
    ax = axes[0,0]
    ax.plot(x, fm, 'C1', lw=2)
    ax.axhline(FM_I, ls=':', color='gray', lw=1, label=f'$f_{{M,i}}={FM_I}$')
    ax.axhline(FM_F, ls='--', color='gray', lw=1, label=f'$f_{{M,f}}={FM_F}$')
    for (label, xv), col in zip(milestones.items(), colors_ms):
        ax.axvline(xv, color=col, ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$x_{loc} = \log_{10}\rho$ (kg/m³)')
    ax.set_ylabel(r'$f_M(x)$')
    ax.set_title('Cohesive Factor (Paper 1 exact)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: fL
    ax = axes[0,1]
    ax.plot(x, fl, 'C0', lw=2)
    for (label, xv), col in zip(milestones.items(), colors_ms):
        ax.axvline(xv, color=col, ls='--', lw=0.8, alpha=0.6,
                   label=label.split('\n')[0])
    ax.set_xlabel(r'$x_{loc} = \log_{10}\rho$ (kg/m³)')
    ax.set_ylabel(r'$f_L(x)$')
    ax.set_title('Entropic Factor (Paper 1 exact)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 3: Screening factor S(x)
    ax = axes[1,0]
    ax.plot(x, screen, 'C2', lw=2)
    ax.axhline(0, ls=':', color='gray', lw=1)
    for xv, col in zip(milestones.values(), colors_ms):
        ax.axvline(xv, color=col, ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$x_{loc} = \log_{10}\rho$ (kg/m³)')
    ax.set_ylabel(r'$S(x) = 1 - f_M(x)/f_{M,i}$')
    ax.set_title('Cohesive Screening Factor')
    ax.grid(alpha=0.3)

    # Panel 4: Total correction factor
    ax = axes[1,1]
    ax.plot(x, corr, 'C3', lw=2.5, label=r'$\alpha \cdot f_L \cdot S(x),\;\alpha=0.5$')
    ax.axhline(0, ls=':', color='gray', lw=1)
    for xv, col in zip(milestones.values(), colors_ms):
        ax.axvline(xv, color=col, ls='--', lw=0.8, alpha=0.6)
    # Mark galactic environments
    environments = {
        'Solar System\n(r~1AU)':         -4.0,
        'Solar neighbourhood\n(r~1kpc)': -20.5,
        'Disc midplane\n(r~5kpc)':       -21.5,
        'Halo outskirts\n(r~30kpc)':     -24.0,
        'Cluster\n(r~1Mpc)':             -25.5,
    }
    for (label, xenv) in environments.items():
        ax.axvline(xenv, color='purple', ls=':', lw=1.2, alpha=0.5)
        ax.text(xenv+0.1, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1]>0 else 0.3,
                label, fontsize=6.5, color='purple', rotation=90, va='top')
    ax.set_xlabel(r'$x_{loc} = \log_{10}\rho$ (kg/m³)')
    ax.set_ylabel('Multiplicative correction')
    ax.set_title(r'GERT Local Correction $\alpha \cdot f_L(x) \cdot S(x)$')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/gert_local/fig1_functions.png', dpi=150)
    plt.close()
    print("Fig 1 saved.")
    return x, fm, fl, screen, corr

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — SYNTHETIC GALAXY TEST
# Galaxy modelled as exponential disc + NFW-equivalent baryonic profile
# No dark matter halo — the GERT correction IS the test
# ─────────────────────────────────────────────────────────────────────────────

G_SI = 6.674e-11   # m³ kg⁻¹ s⁻²
KPC  = 3.0857e19   # m
M_SUN = 1.989e30   # kg
KM_S  = 1e3        # m/s

def exponential_disc_Mb(r_kpc, M_disc_Msun, R_d_kpc):
    """
    Enclosed baryonic mass for an exponential disc:
    Σ(r) = (M_disc / 2πR_d²) exp(-r/R_d)
    M(<r) = M_disc [1 - (1 + r/R_d) exp(-r/R_d)]
    """
    x = r_kpc / R_d_kpc
    return M_disc_Msun * (1.0 - (1.0 + x) * np.exp(-x))

def test_synthetic_galaxy(M_disc_Msun=5e10, R_d_kpc=3.0,
                          alpha_values=[0.3, 0.5, 0.7, 1.0],
                          name="Milky-Way-like"):
    """
    Test the GERT local correction on a synthetic exponential disc galaxy.

    Returns arrays for plotting and numerical verification.
    """
    r_kpc = np.linspace(0.3, 40.0, 500)
    r_m   = r_kpc * KPC

    # Enclosed baryonic mass [kg]
    Mb_Msun = exponential_disc_Mb(r_kpc, M_disc_Msun, R_d_kpc)
    Mb_kg   = Mb_Msun * M_SUN

    # Local density variable [log10 kg/m³]
    x_loc = x_local(r_m, Mb_kg)

    # Baryonic (Newtonian) acceleration [m/s²]
    g_bar_arr = G_SI * Mb_kg / r_m**2

    # Rotation velocities
    v_bar_kms = np.sqrt(np.maximum(g_bar_arr * r_m, 0)) / KM_S

    results = {'r_kpc': r_kpc, 'x_loc': x_loc,
               'g_bar': g_bar_arr, 'v_bar': v_bar_kms,
               'fM_loc': fM(x_loc), 'fL_loc': fL(x_loc),
               'screen': screening_factor(x_loc)}

    for alpha in alpha_values:
        g_gert = g_GERT(g_bar_arr, x_loc, alpha)
        v_gert = np.sqrt(np.maximum(g_gert * r_m, 0)) / KM_S
        results[f'v_gert_a{alpha}'] = v_gert
        results[f'g_gert_a{alpha}'] = g_gert

    return results

def plot_galaxy_test(res, alpha_values, name="Milky-Way-like"):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle(f"GERT Local Test — {name}", fontsize=14, fontweight='bold')

    r   = res['r_kpc']
    x   = res['x_loc']
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

    # Panel 1: Rotation curves
    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(r, res['v_bar'], 'k--', lw=2, label='Newton (baryons only)')
    for alpha, col in zip(alpha_values, colors):
        ax1.plot(r, res[f'v_gert_a{alpha}'], color=col, lw=2,
                 label=fr'GERT local $\alpha={alpha}$')
    ax1.set_xlabel('r (kpc)')
    ax1.set_ylabel('v (km/s)')
    ax1.set_title('Rotation Curves')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 40)

    # Panel 2: Local state variable x_loc(r)
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(r, x, 'k', lw=2)
    for label, xv, col in [
        (r'$\log\rho_{L2}=-23.93$', -23.93, 'C2'),
        (r'$\log\rho_M=-20.30$',    -20.30, 'C1'),
        (r'$\log\rho_c=-17.41$',    -17.41, 'C3'),
        (r'$\log\rho_L=-25.60$',    -25.60, 'C0'),
    ]:
        ax2.axhline(xv, ls='--', color=col, lw=1.2, label=label)
    ax2.set_xlabel('r (kpc)')
    ax2.set_ylabel(r'$x_{loc}(r) = \log_{10}\bar\rho_b(r)$')
    ax2.set_title('Local thermodynamic state variable')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: fL and fM evaluated at x_loc(r)
    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(r, res['fM_loc'], 'C1', lw=2, label=r'$f_M(x_{loc})$')
    ax3.plot(r, res['fL_loc'], 'C0', lw=2, label=r'$f_L(x_{loc})$')
    ax3.plot(r, res['screen'], 'C2--', lw=2, label=r'$S(x_{loc}) = 1-f_M/f_{M,i}$')
    ax3.set_xlabel('r (kpc)')
    ax3.set_ylabel('Factor value')
    ax3.set_title('GERT factors along the galaxy')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Panel 4: Multiplicative correction g_GERT/g_bar
    ax4 = fig.add_subplot(gs[2,0])
    for alpha, col in zip(alpha_values, colors):
        ratio = res[f'g_gert_a{alpha}'] / res['g_bar']
        ax4.plot(r, ratio, color=col, lw=2, label=fr'$\alpha={alpha}$')
    ax4.axhline(1.0, ls=':', color='gray', lw=1)
    ax4.set_xlabel('r (kpc)')
    ax4.set_ylabel(r'$g_{GERT}/g_{bar}$')
    ax4.set_title('Multiplicative correction factor')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # Panel 5: Velocity boost
    ax5 = fig.add_subplot(gs[2,1])
    for alpha, col in zip(alpha_values, colors):
        boost = res[f'v_gert_a{alpha}'] / res['v_bar']
        ax5.plot(r, boost, color=col, lw=2, label=fr'$\alpha={alpha}$')
    ax5.axhline(1.0, ls=':', color='gray', lw=1)
    ax5.set_xlabel('r (kpc)')
    ax5.set_ylabel(r'$v_{GERT}/v_{bar}$')
    ax5.set_title('Velocity boost ratio')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    plt.savefig(f'/home/claude/gert_local/fig2_galaxy_{name.replace(" ","_")}.png',
                dpi=150)
    plt.close()
    print(f"Fig 2 saved for {name}.")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — SOLAR SYSTEM INTEGRITY CHECK (non-falsification criterion 1)
# ─────────────────────────────────────────────────────────────────────────────

def solar_system_check():
    """
    Verify that the GERT local correction is negligible in the Solar System.
    If not → framework is falsified by known observations.
    """
    print("\n" + "="*60)
    print("SOLAR SYSTEM INTEGRITY CHECK")
    print("="*60)

    # Sun mass only within r = 1 AU
    M_sun_kg   = 1.989e30
    r_AU_m     = 1.496e11   # 1 AU in metres
    x_sol      = x_local(r_AU_m, M_sun_kg)
    fM_sol     = fM(x_sol)
    fL_sol     = fL(x_sol)
    screen_sol = screening_factor(x_sol)
    corr_sol   = 0.5 * fL_sol * screen_sol   # α=0.5

    print(f"  r = 1 AU,  M(<r) = M_sun")
    print(f"  x_loc                    = {x_sol:.4f}  (log10 kg/m³)")
    print(f"  fM(x_loc)                = {fM_sol:.6f}")
    print(f"  fL(x_loc)                = {fL_sol:.6f}")
    print(f"  S(x_loc) = 1 - fM/fM_i  = {screen_sol:.6f}")
    print(f"  Correction (α=0.5)       = {corr_sol:.2e}  ({corr_sol*100:.4f}%)")

    if abs(corr_sol) < 1e-6:
        print("  ✅ PASS — Solar System correction is negligible (<1 ppm)")
    elif abs(corr_sol) < 1e-4:
        print("  ⚠️  MARGINAL — correction < 0.01% but non-zero")
    else:
        print("  ❌ FAIL — correction too large, framework falsified by Solar System")

    return x_sol, corr_sol

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — DENSITY SCALE CONCORDANCE CHECK (non-falsification criterion 2)
# ─────────────────────────────────────────────────────────────────────────────

def density_concordance_check():
    """
    Verify that the density scales where the GERT correction peaks
    correspond to the galactic scales where dark matter is most needed.
    This is the key physical alignment argument.
    """
    print("\n" + "="*60)
    print("DENSITY SCALE CONCORDANCE CHECK")
    print("="*60)

    environments = {
        'Solar System (r=1AU, M=M☉)':        (1.496e11,  1.989e30),
        'Molecular cloud (r=1pc, M=1e4 M☉)': (3.086e16,  1.989e34),
        'Disc midplane (r=3kpc, M=3e10 M☉)': (9.26e19,   1.989e40),
        'Solar neigh. (r=1kpc, M=5e9 M☉)':   (3.086e19,  1.989e39),
        'Disc outer (r=10kpc, M=5e10 M☉)':   (3.086e20,  1.989e40),
        'Halo 20kpc  (r=20kpc, M=8e10 M☉)':  (6.17e20,   1.989e41*0.8),
        'Halo 30kpc  (r=30kpc, M=1e11 M☉)':  (9.26e20,   1.989e41),
        'Cluster     (r=1Mpc,  M=1e14 M☉)':  (3.086e22,  1.989e44),
    }

    print(f"  {'Environment':<40} {'x_loc':>8} {'fL':>6} {'S(x)':>6} {'Corr(α=0.5)':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*6} {'-'*12}")

    results = {}
    for name, (r_m, Mb_kg) in environments.items():
        xl   = x_local(r_m, Mb_kg)
        fl_v = fL(xl)
        sc_v = screening_factor(xl)
        corr = 0.5 * fl_v * sc_v
        results[name] = (xl, fl_v, sc_v, corr)
        flag = ""
        if xl > -18:   flag = "← Solar System regime (correction→0)"
        if -24.5 < xl < -23.0: flag = "← PEAK CORRECTION (dark matter regime!)"
        print(f"  {name:<40} {xl:>8.2f} {fl_v:>6.3f} {sc_v:>6.3f} {corr:>12.4f}  {flag}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — GERT vs MOND DISCRIMINATION TEST
# Key prediction: GERT threshold is in DENSITY; MOND threshold is in ACCELERATION
# ─────────────────────────────────────────────────────────────────────────────

def gert_vs_mond_discrimination():
    """
    Construct two hypothetical galaxies that GERT and MOND predict differently:
    Case A: Compact massive galaxy — low density in outskirts but low acceleration
    Case B: Diffuse low-mass galaxy — density crosses x_loc ~ -23.93 early
    """
    print("\n" + "="*60)
    print("GERT vs MOND DISCRIMINATION TEST")
    print("="*60)
    a0_MOND = 1.2e-10   # m/s² — MOND acceleration scale

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GERT vs MOND Discrimination\n"
                 "GERT threshold: density | MOND threshold: acceleration",
                 fontsize=12, fontweight='bold')

    galaxy_cases = [
        # (name, M_disc_Msun, R_d_kpc, description)
        ("Compact Massive\n(M=2×10¹¹ M☉, Rd=1.5kpc)",
         2e11, 1.5, "High mass, compact: g>a0 but x_loc crosses -23.93"),
        ("Diffuse Dwarf\n(M=5×10⁸ M☉,  Rd=4.0kpc)",
         5e8,  4.0, "Low mass, diffuse: g<a0 but x_loc stays above -25"),
    ]

    for idx, (name, Md, Rd, desc) in enumerate(galaxy_cases):
        ax = axes[idx]
        r_kpc = np.linspace(0.3, 30.0, 400)
        r_m   = r_kpc * KPC
        Mb_kg = exponential_disc_Mb(r_kpc, Md, Rd) * M_SUN
        x_loc = x_local(r_m, Mb_kg)
        g_bar = G_SI * Mb_kg / r_m**2
        v_bar = np.sqrt(np.maximum(g_bar * r_m, 0)) / KM_S

        # GERT prediction (α=0.5)
        g_gert = g_GERT(g_bar, x_loc, 0.5)
        v_gert = np.sqrt(np.maximum(g_gert * r_m, 0)) / KM_S

        # Simple MOND (interpolation function μ = x/(1+x))
        nu = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * a0_MOND / np.maximum(g_bar, 1e-20)))
        g_mond = nu * g_bar
        v_mond = np.sqrt(np.maximum(g_mond * r_m, 0)) / KM_S

        ax.plot(r_kpc, v_bar,  'k--', lw=2, label='Newton (baryons)')
        ax.plot(r_kpc, v_gert, 'C3',  lw=2.5, label=r'GERT local ($\alpha=0.5$)')
        ax.plot(r_kpc, v_mond, 'C0',  lw=2.5, ls='-.', label='MOND (simple)')

        # Mark where g_bar = a0 (MOND threshold)
        mond_thresh = np.argmin(np.abs(g_bar - a0_MOND))
        ax.axvline(r_kpc[mond_thresh], color='C0', ls=':', lw=1.5,
                   label=f'MOND thresh: r={r_kpc[mond_thresh]:.1f} kpc')

        # Mark where x_loc = log_rho_L2 (GERT threshold)
        gert_thresh = np.argmin(np.abs(x_loc - LOG_RHO_L2))
        ax.axvline(r_kpc[gert_thresh], color='C3', ls=':', lw=1.5,
                   label=fr'GERT thresh: r={r_kpc[gert_thresh]:.1f} kpc')

        ax.set_xlabel('r (kpc)')
        ax.set_ylabel('v (km/s)')
        ax.set_title(f'{name}\n{desc}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 30)

        print(f"\n  {name.replace(chr(10),' ')}")
        print(f"    MOND threshold at r = {r_kpc[mond_thresh]:.1f} kpc  (g_bar = a0)")
        print(f"    GERT threshold at r = {r_kpc[gert_thresh]:.1f} kpc  (x_loc = log_rho_L2)")
        diff = r_kpc[gert_thresh] - r_kpc[mond_thresh]
        if abs(diff) > 1.0:
            print(f"    → Δr = {diff:+.1f} kpc: GERT and MOND predict DIFFERENT onset radii ✅ discriminable")
        else:
            print(f"    → Δr = {diff:+.1f} kpc: thresholds coincide (degenerate in this case)")

    plt.tight_layout()
    plt.savefig('/home/claude/gert_local/fig3_gert_vs_mond.png', dpi=150)
    plt.close()
    print("\n  Fig 3 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs('/home/claude/gert_local', exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   GERT LOCAL MINI-FRAMEWORK — Internal Test v0.1        ║")
    print("║   Using EXACT Paper 1 parameters — zero new functions    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Block 3: Plot functions
    print("\n[Block 3] Plotting fM, fL across galactic density range...")
    plot_functions_local_range()

    # Block 5: Solar System check (non-falsification criterion)
    solar_system_check()

    # Block 6: Density concordance
    density_concordance_check()

    # Block 4: Synthetic galaxy test
    print("\n[Block 4] Running synthetic galaxy rotation curve test...")
    alphas = [0.3, 0.5, 0.7, 1.0]
    res_mw = test_synthetic_galaxy(
        M_disc_Msun=5e10, R_d_kpc=3.0,
        alpha_values=alphas, name="MW-like"
    )
    plot_galaxy_test(res_mw, alphas, "MW-like (5×10¹⁰ M☉, Rd=3kpc)")

    res_dw = test_synthetic_galaxy(
        M_disc_Msun=1e9, R_d_kpc=2.0,
        alpha_values=alphas, name="Dwarf"
    )
    plot_galaxy_test(res_dw, alphas, "Dwarf (10⁹ M☉, Rd=2kpc)")

    # Block 7: GERT vs MOND discrimination
    print("\n[Block 7] GERT vs MOND discrimination test...")
    gert_vs_mond_discrimination()

    print("\n" + "="*60)
    print("SUMMARY TABLE — Velocity boost at r=30kpc (α=0.5)")
    print("="*60)
    for name, res in [("MW-like", res_mw), ("Dwarf", res_dw)]:
        idx = np.argmin(np.abs(res['r_kpc'] - 30.0))
        vbar = res['v_bar'][idx]
        vgert= res['v_gert_a0.5'][idx]
        xloc = res['x_loc'][idx]
        print(f"  {name:<10}  v_bar={vbar:.1f} km/s  v_GERT={vgert:.1f} km/s  "
              f"boost={vgert/vbar:.2f}x  x_loc={xloc:.2f}")

    print("\n✅ All tests complete. Figures saved to /home/claude/gert_local/")
