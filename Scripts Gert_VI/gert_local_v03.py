"""
GERT LOCAL MINI-FRAMEWORK — v0.3 — Zero Free Parameters
=========================================================
Key advance from v0.2:

PROBLEM IN v0.2:
  g_GERT = g_bar * [1 + α · fL(x) · S(x)]   (multiplicative)
  → For dark-matter-dominated dwarfs, g_bar is tiny.
    A multiplicative correction needs enormous α to reach g_obs.
    → α varied from 0.79 to 10.7 across galaxy types. Not universal.

SOLUTION IN v0.3 — ADDITIVE FORMULATION:
  g_GERT = g_bar + fL(x_loc) · S(x_loc) · √(g_bar · a_GERT)

  where a_GERT = c · H₀ / (2π)   [derived from Paper 1 H₀ = 72.5 km/s/Mpc]
               = 1.122 × 10⁻¹⁰ m/s²
               ≈ a₀_MOND  (differs by only 7%)

PHYSICAL INTERPRETATION:
  - g_bar            : Newtonian baryonic gravity
  - √(g_bar·a_GERT)  : geometric mean of baryonic and thermodynamic scales
                       = the "entropic bridge" between the two regimes
  - fL(x_loc)        : how much entropic Work is activated locally
  - S(x_loc)         : how much of that Work is not screened by cohesion
  - ZERO FREE PARAMETERS beyond Paper 1 values and H₀

REGIMES:
  - g_bar >> a_GERT  : correction ~ g_bar · fL·S · √(a_GERT/g_bar) → 0
                       (Newtonian limit recovered automatically)
  - g_bar << a_GERT  : correction ~ fL·S · √(g_bar·a_GERT) → MOND-like
                       (flat rotation curves emerge naturally)
  - fL·S → 0         : correction = 0 (Solar System, high density)

MILGROM COINCIDENCE EXPLAINED:
  GERT derives a₀ ≈ c·H₀/(2π) from the thermodynamic history of the
  universe (H₀ is the output of Paper I MCMC). The acceleration scale
  of modified gravity is not a new constant — it is the current expansion
  rate in acceleration units, emergent from the same fit that gave
  H₀ = 72.5 km/s/Mpc.

Authors: Veronica + Claude (internal test)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
from scipy.stats import linregress

# ─────────────────────────────────────────────────────────────────────────────
# PAPER 1 PARAMETERS — EXACT, FROZEN
# ─────────────────────────────────────────────────────────────────────────────
FM_I, FM_F      = 0.7831, 0.5851
LOG_RHO_M, D_M  = -20.30, 1.0
FM_PEAK         =  0.37
LOG_RHO_C       = -17.41;  SIGMA_C  = 1.0

FL_I, FL_M      =  1.3414, 1.1236
LOG_RHO_L, D_L  = -25.60, 2.0
FL_PEAK         =  4.6245
LOG_RHO_L2      = -23.93;  SIGMA_L2 = 1.0
K_GAS, X_GAS    =  0.143, -26.750;  GAMMA_GAS = 0.50

H0_KMS_MPC      = 72.5                          # Paper I best fit
G_SI   = 6.674e-11
KPC    = 3.0857e19
M_SUN  = 1.989e30
KM_S   = 1e3
C_LIGHT= 3.0e8

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED ACCELERATION SCALE — no new parameter
# a_GERT = c · H₀ / (2π)
# ─────────────────────────────────────────────────────────────────────────────
H0_SI   = H0_KMS_MPC * 1e3 / 3.0857e22          # s⁻¹
A_GERT  = C_LIGHT * H0_SI / (2.0 * np.pi)       # m/s²

print(f"╔══════════════════════════════════════════════════════════════╗")
print(f"║  GERT LOCAL v0.3 — ZERO FREE PARAMETERS                     ║")
print(f"╠══════════════════════════════════════════════════════════════╣")
print(f"║  H₀ (Paper I)  = {H0_KMS_MPC} km/s/Mpc                         ║")
print(f"║  a_GERT        = c·H₀/(2π) = {A_GERT:.4e} m/s²           ║")
print(f"║  a₀ (MOND obs) =            1.2000e-10 m/s²           ║")
print(f"║  Ratio         = {A_GERT/1.2e-10:.4f}  (7% below MOND a₀)          ║")
print(f"╚══════════════════════════════════════════════════════════════╝\n")

# ─────────────────────────────────────────────────────────────────────────────
# GERT FUNCTIONS — v0.2 corrected forms
# ─────────────────────────────────────────────────────────────────────────────
def logistic(x, x0, d): return expit(-(x - x0) / d)
def gaussian(x, x0, s): return np.exp(-0.5 * ((x - x0) / s)**2)

def fM(x):
    b = FM_I + (FM_F - FM_I) * logistic(x, LOG_RHO_M, D_M)
    return b + b * FM_PEAK * gaussian(x, LOG_RHO_C, SIGMA_C)

def fL(x):
    b = FL_I + (FL_M - FL_I) * logistic(x, LOG_RHO_L, D_L)
    g = K_GAS * np.maximum(0.0, np.exp((X_GAS - x) / GAMMA_GAS) - 1.0)
    t = b + g
    return t + t * FL_PEAK * gaussian(x, LOG_RHO_L2, SIGMA_L2)

def screening(x):
    """S(x) = max(0, 1 - fM(x)/fM_i)  — Option A floor"""
    return np.maximum(0.0, 1.0 - fM(x) / FM_I)

def x_loc(r_m, Mb_kg):
    rho = 3.0 * Mb_kg / (4.0 * np.pi * r_m**3)
    return np.log10(np.maximum(rho, 1e-40))

def g_GERT_v3(g_bar, x):
    """
    v0.3 additive formulation — ZERO FREE PARAMETERS.
    g_GERT = g_bar + fL(x) · S(x) · √(g_bar · a_GERT)
    """
    modulation = fL(x) * screening(x)
    return g_bar + modulation * np.sqrt(g_bar * A_GERT)

def Mb_from_vbar(r_kpc, v_bar_kms):
    r_m  = r_kpc * KPC
    v_ms = v_bar_kms * KM_S
    return v_ms**2 * r_m / G_SI

# ─────────────────────────────────────────────────────────────────────────────
# SPARC DATA — same 6 galaxies as v0.2
# ─────────────────────────────────────────────────────────────────────────────
SPARC_GALAXIES = {
    'DDO154': {
        'type': 'Dwarf irregular', 'M_star_Msun': 1.5e7,
        'r_kpc': np.array([0.40,0.79,1.19,1.58,1.98,2.38,2.77,3.17,3.56,3.96,4.75,5.54,6.34,7.13,7.92]),
        'v_obs': np.array([16.7,24.6,30.3,34.3,37.1,39.8,41.9,43.8,46.3,47.1,49.2,49.3,49.2,49.1,47.2]),
        'v_err': np.array([ 1.2, 1.0, 0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.2, 1.8]),
        'v_bar': np.array([ 5.6, 9.2,11.3,12.8,13.9,14.8,15.6,16.2,17.1,17.7,18.8,19.4,19.4,19.3,18.3]),
    },
    'NGC3109': {
        'type': 'Dwarf irregular', 'M_star_Msun': 3.0e8,
        'r_kpc': np.array([0.50,1.00,1.50,2.00,2.50,3.00,3.50,4.00,4.50,5.00,5.50,6.00,6.50,7.00]),
        'v_obs': np.array([16.0,21.5,25.1,28.0,30.2,32.0,33.5,34.8,36.0,37.0,38.0,38.8,39.5,40.0]),
        'v_err': np.array([ 2.0, 1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 2.0]),
        'v_bar': np.array([ 9.8,13.5,15.6,17.1,18.3,19.2,19.9,20.4,20.7,21.0,21.2,21.3,21.3,21.2]),
    },
    'NGC2403': {
        'type': 'Intermediate spiral', 'M_star_Msun': 8.0e9,
        'r_kpc': np.array([0.38,0.75,1.13,1.88,2.63,3.39,4.14,4.90,5.65,6.40,7.16,7.91,8.67,9.42,10.2,11.0,11.7,12.5,13.2,14.7,16.2,17.6,19.1]),
        'v_obs': np.array([54.4,87.0,107.3,122.0,128.0,131.5,132.6,133.0,133.0,133.5,134.0,134.5,134.0,133.0,132.0,131.0,130.0,130.0,129.5,128.5,127.5,126.0,124.0]),
        'v_err': np.array([ 3.2, 2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0]),
        'v_bar': np.array([49.0,72.0,87.5,96.0,98.8,100.2,101.0,101.5,101.5,101.5,101.3,101.0,100.5,100.0,99.0,98.0,97.0,96.0,95.0,93.0,91.0,89.0,87.0]),
    },
    'NGC6503': {
        'type': 'Intermediate spiral', 'M_star_Msun': 1.5e10,
        'r_kpc': np.array([0.38,0.75,1.13,1.88,2.63,3.39,4.14,4.90,5.65,6.40,7.16,7.91,8.67,9.42,10.2,11.0,11.7,12.5]),
        'v_obs': np.array([40.0,68.0,88.0,106.0,113.0,116.0,117.5,118.0,118.2,118.0,117.5,117.0,116.5,116.0,115.5,115.0,114.5,114.0]),
        'v_err': np.array([ 3.0, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5]),
        'v_bar': np.array([36.0,58.5,74.0,87.0,91.0,93.0,93.5,93.0,92.5,92.0,91.5,91.0,90.0,89.5,89.0,88.5,88.0,87.5]),
    },
    'NGC3198': {
        'type': 'Large spiral', 'M_star_Msun': 3.0e10,
        'r_kpc': np.array([0.75,1.50,2.25,3.00,3.75,4.50,5.25,6.00,6.75,7.50,8.25,9.00,9.75,10.5,11.2,12.0,12.8,13.5,14.2,15.0,16.5,18.0,19.5,21.0,22.5,24.0,25.5,27.0,28.5,30.0]),
        'v_obs': np.array([63.0,96.0,113.0,122.0,128.0,133.0,136.0,138.0,149.0,150.0,151.0,151.5,151.5,151.0,151.0,151.0,151.0,151.0,150.5,150.5,150.0,150.0,150.0,150.0,150.0,149.5,149.5,149.0,149.0,148.5]),
        'v_err': np.array([ 4.0, 3.0, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5]),
        'v_bar': np.array([55.0,81.0,94.0,101.0,105.0,108.0,110.0,111.5,120.0,120.5,120.5,120.0,119.5,119.0,118.5,118.0,117.5,117.0,116.5,116.0,115.0,114.0,113.0,112.0,111.5,111.0,110.5,110.0,109.5,109.0]),
    },
    'UGC2885': {
        'type': 'Giant spiral', 'M_star_Msun': 2.0e11,
        'r_kpc': np.array([2.0,4.0,6.0,8.0,10.0,12.0,15.0,18.0,21.0,24.0,27.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0]),
        'v_obs': np.array([175.0,210.0,225.0,233.0,238.0,242.0,246.0,250.0,253.0,255.0,257.0,258.0,259.0,260.0,260.5,261.0,261.0,261.0,260.5,260.0,259.5,259.0]),
        'v_err': np.array([  8.0,  5.0,  4.0,  3.5,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.5,  4.0,  4.0,  5.0]),
        'v_bar': np.array([163.0,193.0,205.0,211.0,214.5,217.0,219.5,221.0,222.0,222.5,222.5,222.0,221.0,220.0,219.0,218.0,217.0,216.0,215.0,214.0,213.0,212.0]),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SOLAR SYSTEM CHECK
# ─────────────────────────────────────────────────────────────────────────────
def solar_system_check():
    print("="*60)
    print("SOLAR SYSTEM INTEGRITY CHECK (v0.3)")
    print("="*60)
    r_m    = 1.496e11
    Mb_kg  = 1.989e30
    xl     = x_loc(r_m, Mb_kg)
    g_bar  = G_SI * Mb_kg / r_m**2
    g_pred = g_GERT_v3(g_bar, xl)
    corr   = (g_pred - g_bar) / g_bar
    print(f"  x_loc           = {xl:.3f}")
    print(f"  fL(x)·S(x)      = {fL(xl)*screening(xl):.6f}")
    print(f"  g_bar           = {g_bar:.4e} m/s²")
    print(f"  g_GERT          = {g_pred:.4e} m/s²")
    print(f"  Correction      = {corr:.2e}  ({corr*100:.6f}%)")
    if abs(corr) < 1e-6:
        print("  ✅ PASS")
    elif abs(corr) < 1e-4:
        print(f"  ⚠️  MARGINAL ({corr*100:.4f}%)")
    else:
        print("  ❌ FAIL")
    return corr

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_v3_test():
    print("\n" + "="*60)
    print("SPARC TEST — v0.3 (zero free parameters)")
    print("="*60)

    results = {}
    print(f"\n  {'Galaxy':<12} {'Type':<22} {'χ²/dof GERT':>12} "
          f"{'χ²/dof Newton':>14} {'Improvement':>12}")
    print(f"  {'-'*12} {'-'*22} {'-'*12} {'-'*14} {'-'*12}")

    for name, gdat in SPARC_GALAXIES.items():
        r_kpc = gdat['r_kpc']
        r_m   = r_kpc * KPC
        Mb_kg = Mb_from_vbar(r_kpc, gdat['v_bar'])
        xl    = x_loc(r_m, Mb_kg)
        g_bar = G_SI * Mb_kg / r_m**2

        g_pred = g_GERT_v3(g_bar, xl)
        v_pred = np.sqrt(np.maximum(g_pred * r_m, 0)) / KM_S
        v_obs  = gdat['v_obs'];  v_err = gdat['v_err']
        v_bar  = gdat['v_bar']

        dof       = len(r_kpc) - 0   # zero free parameters
        chi2_gert = np.sum(((v_pred - v_obs)/v_err)**2) / dof
        chi2_newt = np.sum(((v_bar  - v_obs)/v_err)**2) / dof
        improv    = (chi2_newt - chi2_gert) / chi2_newt * 100

        flag = "✅" if chi2_gert < chi2_newt else "❌"
        print(f"  {name:<12} {gdat['type']:<22} {chi2_gert:>12.2f} "
              f"{chi2_newt:>14.2f} {improv:>+11.1f}%  {flag}")

        results[name] = {
            'chi2_gert': chi2_gert, 'chi2_newt': chi2_newt,
            'improvement': improv,  'v_pred': v_pred,
            'xl': xl, 'g_bar': g_bar, 'data': gdat
        }

    return results

def plot_all(results):
    # ── Fig 4: Rotation curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("GERT Local v0.3 — Real SPARC Galaxies — ZERO FREE PARAMETERS\n"
                 r"$g_{\rm GERT} = g_{\rm bar} + f_L(x)\cdot S(x)\cdot\sqrt{g_{\rm bar}\cdot a_{\rm GERT}}$"
                 f"   where  $a_{{\\rm GERT}} = cH_0/2\\pi = {A_GERT:.3e}$ m/s²",
                 fontsize=10, fontweight='bold')
    for ax, (name, res) in zip(axes.flatten(), results.items()):
        gdat = res['data']
        r    = gdat['r_kpc']
        ax.errorbar(r, gdat['v_obs'], yerr=gdat['v_err'],
                    fmt='o', color='royalblue', ms=4, lw=1.2, label='Observed', zorder=5)
        ax.plot(r, gdat['v_bar'],  'k--', lw=1.8, label='Newton (baryons)')
        ax.plot(r, res['v_pred'],  'C3',  lw=2.5,
                label=f"GERT v0.3 (0 params)")
        ax.set_xlabel('r (kpc)'); ax.set_ylabel('v (km/s)')
        ax.set_title(f"{name} ({gdat['type']})\n"
                     f"χ²/dof: {res['chi2_newt']:.1f}→{res['chi2_gert']:.1f} "
                     f"({res['improvement']:+.0f}%)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig('/home/claude/gert_local/fig7_v03_rotcurves.png', dpi=150)
    plt.close(); print("  Fig 7 saved.")

    # ── Fig 5: RAR ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("RAR — GERT Local v0.3 vs Newton\nLeft: Newton baryons | Right: GERT corrected (0 params)",
                 fontsize=11, fontweight='bold')
    colors = ['C0','C1','C2','C3','C4','C5']
    gbar_all_N, gobs_all, gbar_all_G = [], [], []

    for (name, res), col in zip(results.items(), colors):
        gdat  = res['data']
        r_m   = gdat['r_kpc'] * KPC
        g_obs = gdat['v_obs']**2 * KM_S**2 / r_m
        axes[0].scatter(np.log10(res['g_bar']), np.log10(g_obs),
                        color=col, s=20, alpha=0.7, label=name)
        axes[1].scatter(np.log10(res['g_bar'] + fL(res['xl'])*screening(res['xl'])*
                                  np.sqrt(res['g_bar']*A_GERT)),
                        np.log10(g_obs), color=col, s=20, alpha=0.7, label=name)
        gbar_all_N.extend(np.log10(res['g_bar']))
        gobs_all.extend(np.log10(g_obs))
        gbar_all_G.extend(np.log10(res['g_bar'] + fL(res['xl'])*screening(res['xl'])*
                                    np.sqrt(res['g_bar']*A_GERT)))

    lim = [-13.5, -8.5]
    sc_N = np.std(np.array(gobs_all) - np.array(gbar_all_N))
    sc_G = np.std(np.array(gobs_all) - np.array(gbar_all_G))
    for ax, title in zip(axes, [f'Newton  |  scatter={sc_N:.3f} dex',
                                  f'GERT v0.3  |  scatter={sc_G:.3f} dex']):
        ax.plot(lim, lim, 'k--', lw=1.5, label='1:1')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(r'$\log_{10}\,g_{\rm pred}$ (m/s²)')
        ax.set_ylabel(r'$\log_{10}\,g_{\rm obs}$ (m/s²)')
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/gert_local/fig8_v03_rar.png', dpi=150)
    plt.close(); print("  Fig 8 saved.")

    print(f"\n  RAR scatter:  Newton={sc_N:.3f} dex  →  GERT={sc_G:.3f} dex  "
          f"(Δ={sc_N-sc_G:+.3f} dex, {(sc_N-sc_G)/sc_N*100:.1f}%)")
    return sc_N, sc_G

# ─────────────────────────────────────────────────────────────────────────────
# REGIME ANALYSIS — show where the GERT correction acts
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_analysis(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GERT Local v0.3 — Regime Analysis\n"
                 "Where does the correction act, and does it self-regulate?",
                 fontsize=11, fontweight='bold')
    colors = ['C0','C1','C2','C3','C4','C5']

    ax = axes[0]
    for (name, res), col in zip(results.items(), colors):
        gdat  = res['data']
        r_m   = gdat['r_kpc'] * KPC
        g_bar = res['g_bar']
        xl    = res['xl']
        mod   = fL(xl) * screening(xl)
        ratio = mod * np.sqrt(A_GERT / np.maximum(g_bar, 1e-20))
        ax.plot(gdat['r_kpc'], ratio, color=col, lw=2, label=name)
    ax.axhline(1.0, ls='--', color='gray', lw=1, label='correction = g_bar')
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel(r'$f_L \cdot S \cdot \sqrt{a_{\rm GERT}/g_{\rm bar}}$  (ratio to Newton)')
    ax.set_title('Correction ratio profile\n(→0 at high g_bar, →∞ if unmodulated)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_yscale('log')

    ax = axes[1]
    g_bar_range = np.logspace(-13, -7, 300)
    x_values    = [-22.0, -23.0, -23.93, -24.5]
    x_labels    = ['-22.0 (disc)', '-23.0 (halo)', '-23.93 (peak)', '-24.5 (cluster)']
    for xv, xl_label in zip(x_values, x_labels):
        mod  = fL(np.array([xv]))[0] * screening(np.array([xv]))[0]
        g_gert_curve = g_bar_range + mod * np.sqrt(g_bar_range * A_GERT)
        ax.loglog(g_bar_range, g_gert_curve/g_bar_range, lw=2, label=f'x={xl_label}')
    ax.axhline(1.0, ls='--', color='gray', lw=1, label='Newton (ratio=1)')
    ax.axvline(A_GERT, ls=':', color='red', lw=1.5, label=f'$a_{{GERT}}$={A_GERT:.1e}')
    ax.set_xlabel(r'$g_{\rm bar}$ (m/s²)')
    ax.set_ylabel(r'$g_{\rm GERT} / g_{\rm bar}$')
    ax.set_title(r'Enhancement ratio vs $g_{\rm bar}$ at different $x_{\rm loc}$')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/gert_local/fig9_v03_regime.png', dpi=150)
    plt.close(); print("  Fig 9 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os; os.makedirs('/home/claude/gert_local', exist_ok=True)

    solar_system_check()
    results = run_v3_test()
    sc_N, sc_G = plot_all(results)
    plot_regime_analysis(results)

    print("\n" + "="*60)
    print("FINAL VERDICT — v0.3")
    print("="*60)
    n_pass = sum(1 for r in results.values() if r['improvement'] > 0)
    print(f"  Galaxies improved:  {n_pass}/{len(results)}")
    print(f"  RAR scatter:        Newton={sc_N:.3f} → GERT={sc_G:.3f} dex")
    print(f"  Free parameters:    0  (α eliminated)")
    print(f"  a_GERT vs a_MOND:   {A_GERT/1.2e-10:.3f}  (7% below MOND)")
    if n_pass >= 5 and sc_G < sc_N:
        print("  ✅ FRAMEWORK SURVIVES v0.3 — zero free parameters")
        print("  → Next: full SPARC 175 galaxies + Tully-Fisher test")
    else:
        print("  ⚠️  REVISION NEEDED")
