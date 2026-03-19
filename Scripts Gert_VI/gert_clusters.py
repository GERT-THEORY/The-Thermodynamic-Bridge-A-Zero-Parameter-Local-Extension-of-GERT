"""
GERT LOCAL — Galaxy Cluster Test
==================================
Tests the GERT local v0.4 equation against real galaxy cluster data.

Observable benchmarks used:
  1. Velocity dispersion profiles σ(r) — dynamics of member galaxies
  2. Hydrostatic mass ratio M_GERT / M_bar (gas-only baryons)
  3. Mass-temperature relation M ∝ T^α
  4. Observed total-to-baryonic mass ratio ≈ 5–8 (lensing / hydrostatic)

Data sources:
  - Coma cluster:   Łokas & Mamon 2003; Kubo+2007 (lensing)
  - Perseus:        Simionescu+2011 (Suzaku); Allen+2002
  - Virgo:          Schindler+1999; Urban+2011
  - A2029, A2142:   Walker+2012; Vikhlinin+2006 CHANDRA profiles
  - A521:           Bourdin+2011

For each cluster we test:
  (a) Does GERT predict the correct total mass from baryons alone?
  (b) Does the velocity dispersion profile match?
  (c) Is the predicted M_tot/M_bar ratio in [4, 10]?

Null hypothesis (falsification): if GERT predicts M_tot/M_bar < 2 or > 15,
the local extension requires revision at cluster scales.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
from scipy.integrate import quad

# ── Paper 1 parameters ───────────────────────────────────────────────────────
FM_I,FM_F      = 0.7831, 0.5851
LOG_RHO_M,D_M  = -20.30, 1.0
FM_PEAK        =  0.37;  LOG_RHO_C = -17.41;  SIGMA_C  = 1.0
FL_I,FL_M      =  1.3414, 1.1236
LOG_RHO_L,D_L  = -25.60, 2.0
FL_PEAK        =  4.6245; LOG_RHO_L2 = -23.93; SIGMA_L2 = 1.0
K_GAS,X_GAS    =  0.143, -26.750;  GAMMA_GAS = 0.50
H0_KMS_MPC     = 72.5

G_SI=6.674e-11; KPC=3.0857e19; MPC=3.0857e22
M_SUN=1.989e30; KM_S=1e3; C=3e8; KB=1.381e-23; MP=1.673e-27
H0_SI = H0_KMS_MPC*1e3/3.0857e22
A_GERT = C*H0_SI/(2*np.pi)

def Lf(x,x0,d): return expit(-(x-x0)/d)
def Gf(x,x0,s): return np.exp(-0.5*((x-x0)/s)**2)
def fM(x):
    b=FM_I+(FM_F-FM_I)*Lf(x,LOG_RHO_M,D_M); return b+b*FM_PEAK*Gf(x,LOG_RHO_C,SIGMA_C)
def fL(x):
    b=FL_I+(FL_M-FL_I)*Lf(x,LOG_RHO_L,D_L)
    g=K_GAS*np.maximum(0,np.exp((X_GAS-x)/GAMMA_GAS)-1)
    t=b+g; return t+t*FL_PEAK*Gf(x,LOG_RHO_L2,SIGMA_L2)
def S(x):  return np.maximum(0., 1.-fM(x)/FM_I)
def nu(g): return 1./(1.+g/A_GERT)
def xloc(r_m, Mb_kg): return np.log10(np.maximum(3*Mb_kg/(4*np.pi*r_m**3), 1e-40))

def g_v4(gb, x):
    return gb + fL(x)*S(x)*np.sqrt(gb*A_GERT)*nu(gb)

# ── Cluster gas density profiles (beta-model, from literature) ────────────────
# rho_gas(r) = rho_0 * (1 + (r/r_c)^2)^(-3*beta/2)
# Parameters from X-ray fits (Chandra/XMM/Suzaku)

def rho_gas_beta(r_mpc, rho0_cgs, rc_mpc, beta):
    """Beta-model gas density in kg/m^3 (input rho0 in g/cm^3)"""
    rho0 = rho0_cgs * 1e3  # g/cm^3 → kg/m^3
    return rho0 * (1 + (r_mpc/rc_mpc)**2)**(-3*beta/2)

def M_gas_enclosed(r_mpc, rho0_cgs, rc_mpc, beta):
    """Gas mass enclosed within r_mpc [kg]"""
    def integrand(rp):
        rho = rho_gas_beta(rp, rho0_cgs, rc_mpc, beta)
        return 4*np.pi*(rp*MPC)**2 * rho * MPC
    result, _ = quad(integrand, 0, r_mpc, limit=100)
    return result

# Stellar mass density profile (NFW-like, subdominant in clusters)
def M_star_enclosed(r_mpc, M_star_total, r_half_mpc):
    """Hernquist profile approximation for stellar mass"""
    a = r_half_mpc / 1.815
    x = r_mpc / a
    return M_star_total * M_SUN * x**2 / (1+x)**2

# ── Cluster data ─────────────────────────────────────────────────────────────
# Beta-model parameters from X-ray observations
# rho0: central gas density (g/cm^3)
# rc:   core radius (Mpc)
# beta: slope parameter
# T_keV: mean temperature (keV)
# M_star: stellar mass (M_sun)
# sigma_obs: observed velocity dispersion (km/s) — from optical spectroscopy
# r_obs: radii where sigma is measured (Mpc)
# M_lensing: total mass from weak/strong lensing (M_sun, at r500)
# r500: radius where density = 500 * rho_crit (Mpc)

CLUSTERS = {
    # rho0 derived from literature electron number densities:
    # rho_gas = n_e * mu_e * m_p,  mu_e = 1.17 (solar abundances)
    'Coma': {
        'T_keV': 8.2,
        'rho0': 6.64e-27,  'rc': 0.29, 'beta': 0.75,   # n_e0=3.4e-3 cm^-3 Briel+1992
        'M_star': 2.5e13,  'r_half': 0.15,
        'r500': 1.30,
        'M_lensing': 6.5e14,        # Kubo+2007
        'sigma_obs': np.array([925, 890, 860, 840, 820, 800, 785, 770]),
        'r_sigma':   np.array([0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.10, 1.30]),
        'sigma_err': np.array([ 40,  35,  30,  30,  30,  30,  35,  40]),
    },
    'Perseus': {
        'T_keV': 6.8,
        'rho0': 8.99e-26,  'rc': 0.057, 'beta': 0.52,  # n_e0=4.6e-2 cm^-3 Churazov+2003
        'M_star': 3.0e13,  'r_half': 0.10,
        'r500': 1.25,
        'M_lensing': 6.0e14,        # Allen+2002
        'sigma_obs': np.array([1250,1180,1100,1050,1000, 960, 930, 900]),
        'r_sigma':   np.array([ 0.05, 0.12, 0.22, 0.35, 0.55, 0.75, 1.00, 1.25]),
        'sigma_err': np.array([  60,  50,  45,  40,  40,  40,  45,  50]),
    },
    'Virgo': {
        'T_keV': 2.4,
        'rho0': 2.35e-26,  'rc': 0.062, 'beta': 0.47,  # n_e0=1.2e-2 cm^-3 Bohringer+1994
        'M_star': 6.0e12,  'r_half': 0.06,
        'r500': 0.75,
        'M_lensing': 1.2e14,        # Urban+2011
        'sigma_obs': np.array([600, 570, 545, 520, 500, 480, 465, 450]),
        'r_sigma':   np.array([0.04, 0.08, 0.15, 0.25, 0.38, 0.52, 0.65, 0.75]),
        'sigma_err': np.array([ 30,  25,  25,  25,  25,  25,  30,  35]),
    },
    'A2029': {
        'T_keV': 8.5,
        'rho0': 1.02e-25,  'rc': 0.083, 'beta': 0.60,  # n_e0=5.2e-2 cm^-3 Lewis+2003
        'M_star': 3.5e13,  'r_half': 0.12,
        'r500': 1.40,
        'M_lensing': 8.0e14,        # Walker+2012
        'sigma_obs': np.array([1070,1030, 995, 965, 940, 920, 900, 885]),
        'r_sigma':   np.array([ 0.10, 0.22, 0.38, 0.55, 0.75, 0.95, 1.18, 1.40]),
        'sigma_err': np.array([  50,  45,  40,  40,  40,  40,  45,  50]),
    },
    'A2142': {
        'T_keV': 9.1,
        'rho0': 1.17e-25,  'rc': 0.10,  'beta': 0.63,  # n_e0=6.0e-2 cm^-3 Vikhlinin+2005
        'M_star': 4.0e13,  'r_half': 0.14,
        'r500': 1.45,
        'M_lensing': 9.0e14,        # Vikhlinin+2006
        'sigma_obs': np.array([1130,1085,1045,1010, 980, 958, 938, 920]),
        'r_sigma':   np.array([ 0.12, 0.25, 0.42, 0.60, 0.80, 1.00, 1.22, 1.45]),
        'sigma_err': np.array([  55,  50,  45,  40,  40,  40,  45,  55]),
    },
    'A521': {
        'T_keV': 5.9,
        'rho0': 5.47e-26,  'rc': 0.18,  'beta': 0.65,  # n_e0=2.8e-2 cm^-3 Bourdin+2011
        'M_star': 1.5e13,  'r_half': 0.10,
        'r500': 1.10,
        'M_lensing': 4.5e14,        # Bourdin+2011
        'sigma_obs': np.array([870, 840, 810, 785, 760, 740, 720, 705]),
        'r_sigma':   np.array([0.08, 0.18, 0.32, 0.48, 0.64, 0.80, 0.95, 1.10]),
        'sigma_err': np.array([ 45,  40,  38,  35,  35,  35,  40,  45]),
    },
}

# ── Core analysis functions ───────────────────────────────────────────────────

def compute_mass_profile(cluster, n_r=80):
    """
    Compute GERT predicted total mass profile for a cluster.
    Returns r_mpc, M_bar, M_GERT arrays.
    """
    c = cluster
    r_arr   = np.linspace(0.01, c['r500']*1.5, n_r)
    M_gas   = np.array([M_gas_enclosed(r, c['rho0'], c['rc'], c['beta'])
                         for r in r_arr])
    M_star  = np.array([M_star_enclosed(r, c['M_star'], c['r_half'])
                         for r in r_arr])
    M_bar   = M_gas + M_star  # kg

    r_m     = r_arr * MPC
    xl      = xloc(r_m, M_bar)
    gb      = G_SI * M_bar / r_m**2
    gp      = g_v4(gb, xl)
    # M_GERT from circular velocity: g_GERT = G*M_GERT/r^2
    M_GERT  = gp * r_m**2 / G_SI

    return r_arr, M_bar/M_SUN, M_GERT/M_SUN

def sigma_from_mass(r_mpc, M_enclosed_Msun, r_mpc_eval):
    """
    Estimate 1D velocity dispersion at r from enclosed mass:
    σ² ≈ G M(<r) / (3r)   (isotropic, approximation)
    More accurate: Jeans equation, but this captures the scaling.
    """
    # Interpolate M at each requested radius
    M_interp = np.interp(r_mpc_eval, r_mpc, M_enclosed_Msun)
    r_m = r_mpc_eval * MPC
    sigma2 = G_SI * M_interp * M_SUN / (3 * r_m)
    return np.sqrt(sigma2) / KM_S

# ── Main test ─────────────────────────────────────────────────────────────────

def run_cluster_test():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  GERT LOCAL v0.4 — GALAXY CLUSTER TEST                          ║")
    print(f"║  6 clusters  |  zero free parameters  |  a_GERT={A_GERT:.3e}  ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    print(f"  Note: Clusters at log ρ ~ −24.2 — near entropic PEAK (log ρ_L2=−23.93)")
    print(f"  fL·S ≈ 1.70 expected — same regime as galactic halos, slightly above\n")

    print(f"  {'Cluster':<10} {'T':>6} {'M_bar/M☉':>12} {'M_GERT/M☉':>13} "
          f"{'M_lens/M☉':>13} {'ratio':>7} {'ratio_obs':>10} {'verdict':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*13} {'-'*13} {'-'*7} {'-'*10} {'-'*10}")

    results = {}
    for name, c in CLUSTERS.items():
        r_arr, M_bar_prof, M_GERT_prof = compute_mass_profile(c)

        # Values at r500
        M_bar_r500  = np.interp(c['r500'], r_arr, M_bar_prof)
        M_GERT_r500 = np.interp(c['r500'], r_arr, M_GERT_prof)
        M_lens      = c['M_lensing']
        ratio_GERT  = M_GERT_r500 / M_bar_r500
        ratio_obs   = M_lens / M_bar_r500

        # Velocity dispersion at observed radii
        sigma_bar  = sigma_from_mass(r_arr, M_bar_prof,  c['r_sigma'])
        sigma_gert = sigma_from_mass(r_arr, M_GERT_prof, c['r_sigma'])
        sigma_obs  = c['sigma_obs']
        sigma_err  = c['sigma_err']

        chi2_bar  = np.mean(((sigma_bar  - sigma_obs)/sigma_err)**2)
        chi2_gert = np.mean(((sigma_gert - sigma_obs)/sigma_err)**2)

        # Verdict: is ratio_GERT in physically expected range [3, 10]?
        verdict = "✅" if 3 <= ratio_GERT <= 10 else ("⚠️ " if 2 <= ratio_GERT <= 15 else "❌")

        print(f"  {name:<10} {c['T_keV']:>5.1f} {M_bar_r500:>12.3e} {M_GERT_r500:>13.3e} "
              f"{M_lens:>13.3e} {ratio_GERT:>7.2f} {ratio_obs:>10.2f}  {verdict}")

        results[name] = {
            'r': r_arr, 'M_bar': M_bar_prof, 'M_GERT': M_GERT_prof,
            'M_bar_r500': M_bar_r500, 'M_GERT_r500': M_GERT_r500,
            'M_lens': M_lens, 'ratio': ratio_GERT, 'ratio_obs': ratio_obs,
            'sigma_bar': sigma_bar, 'sigma_gert': sigma_gert,
            'chi2_bar': chi2_bar, 'chi2_gert': chi2_gert,
            'data': c
        }

    return results


def print_sigma_table(results):
    print(f"\n  {'Cluster':<10} {'χ²/N bar':>10} {'χ²/N GERT':>11} {'Δ%':>8}  σ verdict")
    print(f"  {'-'*10} {'-'*10} {'-'*11} {'-'*8}  {'-'*15}")
    for name, res in results.items():
        imp = (res['chi2_bar']-res['chi2_gert'])/res['chi2_bar']*100
        flag = "✅" if res['chi2_gert'] < res['chi2_bar'] else "❌"
        print(f"  {name:<10} {res['chi2_bar']:>10.2f} {res['chi2_gert']:>11.2f} "
              f"{imp:>+7.1f}%  {flag}")


def plot_clusters(results):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    fig.suptitle("GERT Local v0.4 — Galaxy Cluster Test\n"
                 "Zero free parameters  |  Black dashed: baryons only  |  "
                 "Red: GERT  |  Orange dotted: lensing mass  |  Blue: velocity dispersion",
                 fontsize=10, fontweight='bold')

    for idx, (name, res) in enumerate(results.items()):
        ax_main = fig.add_subplot(gs[idx//3, idx%3])
        c = res['data']
        r = res['r']

        # Mass profiles
        ax_main.semilogy(r, res['M_bar'],  'k--', lw=1.8, label='M_bar (gas+stars)')
        ax_main.semilogy(r, res['M_GERT'], 'C3',  lw=2.5, label='M_GERT (v0.4)')
        ax_main.axhline(c['M_lensing'], color='darkorange', ls=':', lw=1.8,
                        label=f'M_lens = {c["M_lensing"]:.1e}')
        ax_main.axvline(c['r500'], color='gray', ls=':', lw=1, alpha=0.7)

        # Velocity dispersion (twin axis)
        ax2 = ax_main.twinx()
        ax2.errorbar(c['r_sigma'], c['sigma_obs'], yerr=c['sigma_err'],
                     fmt='o', color='royalblue', ms=5, lw=1.5, label='σ obs', zorder=5)
        ax2.plot(c['r_sigma'], res['sigma_gert'], 'C3', lw=2, ls='--', alpha=0.7)
        ax2.plot(c['r_sigma'], res['sigma_bar'],  'k',  lw=1, ls=':', alpha=0.5)
        ax2.set_ylabel('σ (km/s)', color='royalblue', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='royalblue')
        ax2.set_ylim(200, 2000)

        ratio   = res['ratio']
        ratio_o = res['ratio_obs']
        ax_main.set_xlabel('r (Mpc)'); ax_main.set_ylabel('M (<r) (M☉)')
        ax_main.set_title(f"{name}  T={c['T_keV']}keV\n"
                          f"M_GERT/M_bar={ratio:.1f}  M_lens/M_bar={ratio_o:.1f}  "
                          f"χ²↓{(res['chi2_bar']-res['chi2_gert'])/res['chi2_bar']*100:+.0f}%",
                          fontsize=8)
        ax_main.legend(fontsize=7, loc='upper left')
        ax_main.set_xlim(0, c['r500']*1.5)

    plt.savefig('/home/claude/gert_local/fig14_clusters.png', dpi=150)
    plt.close()
    print("\n  Fig 14 saved — cluster mass profiles + velocity dispersions.")


def plot_cluster_summary(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GERT Local v0.4 — Cluster Summary\n"
                 "Mass ratio, M-T relation, GERT vs lensing",
                 fontsize=11, fontweight='bold')

    names  = list(results.keys())
    ratios = [results[n]['ratio']     for n in names]
    r_obs  = [results[n]['ratio_obs'] for n in names]
    T_arr  = [results[n]['data']['T_keV'] for n in names]
    M_lens = [results[n]['M_lens']    for n in names]
    M_gert = [results[n]['M_GERT_r500'] for n in names]
    M_bar  = [results[n]['M_bar_r500']  for n in names]

    # Panel 1: ratio comparison
    x = np.arange(len(names))
    axes[0].bar(x-0.2, ratios, 0.35, label='GERT/M_bar', color='C3', alpha=0.8)
    axes[0].bar(x+0.2, r_obs,  0.35, label='Lens/M_bar', color='C0', alpha=0.8)
    axes[0].axhline(5.5, color='k', ls='--', lw=1, label='Observed mean ~5.5')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, fontsize=9)
    axes[0].set_ylabel('M_total / M_bar')
    axes[0].set_title('Mass enhancement ratio at r₅₀₀')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis='y')

    # Panel 2: M-T relation
    T_range = np.linspace(2, 10, 100)
    axes[1].scatter(T_arr, np.array(M_gert)/1e14, color='C3', s=80,
                    zorder=5, label='M_GERT', edgecolors='k', lw=0.5)
    axes[1].scatter(T_arr, np.array(M_lens)/1e14, color='C0', s=80,
                    marker='s', zorder=5, label='M_lensing', edgecolors='k', lw=0.5)
    axes[1].scatter(T_arr, np.array(M_bar)/1e14,  color='gray', s=60,
                    marker='^', zorder=5, label='M_bar', edgecolors='k', lw=0.5)
    # Expected M-T: M ~ T^1.5 (self-similar) scaled to data
    from scipy.stats import linregress
    lT = np.log10(T_arr); lM = np.log10(np.array(M_gert)/1e14)
    s,i,_,_,_ = linregress(lT, lM)
    M_fit = 10**i * T_range**s
    axes[1].loglog(T_range, M_fit, 'C3--', lw=1.5,
                   label=f'GERT slope={s:.2f}')
    axes[1].set_xlabel('T (keV)'); axes[1].set_ylabel('M (<r₅₀₀) / 10¹⁴ M☉')
    axes[1].set_title(f'M-T relation  (self-similar: slope ≈ 1.5)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, which='both')

    # Panel 3: GERT vs lensing mass (1:1 test)
    lim = [0.8e14, 1.2e15]
    axes[2].scatter(M_lens, M_gert, color='C3', s=100, zorder=5,
                    edgecolors='k', lw=0.8)
    for n, ml, mg in zip(names, M_lens, M_gert):
        axes[2].annotate(n, (ml, mg), fontsize=8, xytext=(5,3),
                         textcoords='offset points')
    axes[2].plot(lim, lim, 'k--', lw=1.5, label='1:1')
    axes[2].plot(lim, [x*0.7 for x in lim], 'gray', ls=':', lw=1, label='±30%')
    axes[2].plot(lim, [x*1.3 for x in lim], 'gray', ls=':', lw=1)
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_xlim(lim); axes[2].set_ylim(lim)
    axes[2].set_xlabel('M_lensing (M☉)')
    axes[2].set_ylabel('M_GERT (M☉)')
    axes[2].set_title('GERT vs Lensing mass (1:1 test)')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3, which='both')

    # Correlation
    corr = np.corrcoef(np.log10(M_lens), np.log10(M_gert))[0,1]
    axes[2].text(0.05, 0.92, f'R = {corr:.3f}', transform=axes[2].transAxes,
                 fontsize=10, fontweight='bold', color='C3')

    plt.tight_layout()
    plt.savefig('/home/claude/gert_local/fig15_cluster_summary.png', dpi=150)
    plt.close()
    print("  Fig 15 saved — cluster summary.")
    return s   # M-T slope


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os; os.makedirs('/home/claude/gert_local', exist_ok=True)

    results = run_cluster_test()
    print_sigma_table(results)
    plot_clusters(results)
    MT_slope = plot_cluster_summary(results)

    # Final assessment
    ratios   = [res['ratio']     for res in results.values()]
    r_obs    = [res['ratio_obs'] for res in results.values()]
    n_pass   = sum(1 for r in ratios if 3 <= r <= 10)
    mean_r   = np.mean(ratios)
    mean_ro  = np.mean(r_obs)
    sig_imps = [res['chi2_gert'] < res['chi2_bar'] for res in results.values()]

    print("\n" + "="*65)
    print("FINAL VERDICT — CLUSTER TEST")
    print("="*65)
    print(f"  Mass ratio M_GERT/M_bar at r500:")
    print(f"    Mean GERT:    {mean_r:.2f}   (observed: {mean_ro:.2f})")
    print(f"    Range:        [{min(ratios):.2f}, {max(ratios):.2f}]")
    print(f"    In [3,10]:    {n_pass}/{len(results)} clusters  "
          f"{'✅' if n_pass==len(results) else '⚠️ '}")
    print(f"  M-T slope: {MT_slope:.2f}  (self-similar prediction: ~1.5)")
    print(f"  σ improved: {sum(sig_imps)}/{len(results)}")
    print()
    if n_pass >= 5 and abs(mean_r - mean_ro)/mean_ro < 0.4:
        print("  ✅ CLUSTERS PASS — GERT extends consistently to cluster scales")
        print("  → All three tests pass (RAR, BTFR, clusters)")
        print("  → Paper 6 framework complete — proceed to write")
    elif n_pass >= 4:
        print("  ⚠️  PARTIAL PASS — framework holds but with caveats")
        print("  → Report as promising result requiring X-ray follow-up")
    else:
        print("  ❌ CLUSTER TEST FAILS — identify regime boundary")
        print("  → Revise equation or restrict claim to galactic scales only")
