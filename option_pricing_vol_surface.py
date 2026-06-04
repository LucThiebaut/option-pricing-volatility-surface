import numpy as np  # calcul numérique (vecteurs/matrices)
import matplotlib.colors as mcolors  # outils pour gérer des couleurs
import matplotlib.pyplot as plt  # tracer des graphiques
from matplotlib.widgets import CheckButtons, Slider, Button, TextBox  # widgets UI (cases, sliders, boutons, champs texte)
from matplotlib.animation import FuncAnimation  # animation frame par frame
from math import erf, sqrt, exp, log, pi  # fonctions math utiles (normale, racine, exponentielle, log, pi)
# ==========================================================
# 0) Lois normales : CDF / PDF (version scalaire)
# ==========================================================
def norm_cdf(x: float) -> float:  # fonction de répartition N(0,1)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))  # formule standard via erf

def norm_pdf(x: float) -> float:  # densité N(0,1)
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)  # formule standard

def smooth_surface_2d(Z, w=3):
    """
    Lissage 2D simple par moyenne mobile.
    w = taille du noyau (3 ou 5 suffisent largement)
    """
    pad = w // 2
    Zp = np.pad(Z, ((pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(Z)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            block = Zp[i:i+w, j:j+w]
            out[i, j] = np.mean(block)

    return out
# ==========================================================
# 1) Black–Scholes (Call, q=0) : prix + greeks de base
# ==========================================================
def bs_d1_d2(S: float, K: float, r: float, sigma: float, tau: float):  # calcule d1 et d2
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt(tau))  # définition de d1
    d2 = d1 - sigma * sqrt(tau)  # définition de d2
    return d1, d2  # renvoie les deux

def bs_call_price(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # prix du call
    if tau <= 0.0:  # si maturité atteinte
        return max(S - K, 0.0)  # payoff
    if sigma <= 0.0:  # si vol nulle, modèle dégénère
        forward = S * exp(r * tau)  # forward sous r
        return exp(-r * tau) * max(forward - K, 0.0)  # prix = payoff du forward discounté
    d1, d2 = bs_d1_d2(S, K, r, sigma, tau)  # calcule d1/d2
    return S * norm_cdf(d1) - K * exp(-r * tau) * norm_cdf(d2)  # formule BS call

def bs_call_delta(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # delta du call
    if tau <= 0.0:  # à maturité
        return 1.0 if S > K else 0.0  # delta du payoff
    if sigma <= 0.0:  # vol quasi nulle
        forward = S * exp(r * tau)  # forward
        return 1.0 if forward > K else 0.0  # delta “binaire” selon forward
    d1, _ = bs_d1_d2(S, K, r, sigma, tau)  # calcule d1 (d2 inutile ici)
    return norm_cdf(d1)  # delta call BS

def bs_call_gamma(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # gamma du call
    if tau <= 0.0 or sigma <= 0.0 or S <= 0.0:  # cas limites
        return 0.0  # gamma nul
    d1, _ = bs_d1_d2(S, K, r, sigma, tau)  # calcule d1
    return norm_pdf(d1) / (S * sigma * sqrt(tau))  # formule gamma

def bs_call_vega(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # vega du call
    if tau <= 0.0 or sigma <= 0.0 or S <= 0.0:  # cas limites
        return 0.0  # vega nul
    d1, _ = bs_d1_d2(S, K, r, sigma, tau)  # calcule d1
    return S * norm_pdf(d1) * sqrt(tau)  # formule vega

def bs_call_theta(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # theta (par an) du call
    if tau <= 0.0 or sigma <= 0.0 or S <= 0.0:  # cas limites
        return 0.0  # theta nul
    d1, d2 = bs_d1_d2(S, K, r, sigma, tau)  # calcule d1/d2
    term1 = -(S * norm_pdf(d1) * sigma) / (2.0 * sqrt(tau))  # partie diffusion
    term2 = -r * K * exp(-r * tau) * norm_cdf(d2)  # partie taux
    return term1 + term2  # theta total

# ==========================================================
# 2) Greeks “secondaires” + greeks numérisés
# ==========================================================
def bs_call_vanna(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # vanna (dVega/dS ou dDelta/dSigma)
    if tau <= 0.0 or sigma <= 0.0 or S <= 0.0:  # cas limites
        return 0.0  # vanna nul
    d1, d2 = bs_d1_d2(S, K, r, sigma, tau)  # d1/d2
    return -norm_pdf(d1) * d2 / sigma  # formule vanna (une convention possible)

def bs_call_vomma(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # vomma (dVega/dSigma)
    if tau <= 0.0 or sigma <= 0.0 or S <= 0.0:  # cas limites
        return 0.0  # vomma nul
    d1, d2 = bs_d1_d2(S, K, r, sigma, tau)  # d1/d2
    vega = bs_call_vega(S, K, r, sigma, tau)  # on réutilise vega
    return vega * d1 * d2 / sigma  # formule vomma

def _eps_tau(tau: float) -> float:  # petit pas pour dériver en tau
    if tau <= 0.0:  # si tau invalide
        return 0.0  # pas nul
    e = 1e-4 * max(1.0, tau)  # pas relatif
    return min(e, 0.25 * tau)  # évite de dépasser tau

def _eps_S(S: float) -> float:  # petit pas pour dériver en spot
    return max(1e-6, 1e-4 * max(1.0, abs(S)))  # pas relatif + sécurité

def _eps_sigma(sig: float) -> float:  # petit pas pour dériver en sigma
    return max(1e-8, 1e-4 * max(1e-3, abs(sig)))  # pas relatif + sécurité

def bs_call_charm_num(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # charm = -dDelta/dt (approx)
    e = _eps_tau(tau)  # pas en tau
    if e == 0.0 or tau - e <= 0.0:  # pas possible
        return 0.0  # renvoie 0
    d_now = bs_call_delta(S, K, r, sigma, tau)  # delta aujourd’hui
    d_less = bs_call_delta(S, K, r, sigma, tau - e)  # delta avec tau plus petit (plus proche maturité)
    dD_dTau = (d_less - d_now) / (-e)  # dérivée approx par différence finie
    return -dD_dTau  # signe choisi pour faire “charm” en temps qui passe

def bs_call_color_num(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # color = -dGamma/dt (approx)
    e = _eps_tau(tau)  # pas en tau
    if e == 0.0 or tau - e <= 0.0:  # pas possible
        return 0.0  # renvoie 0
    g_now = bs_call_gamma(S, K, r, sigma, tau)  # gamma actuel
    g_less = bs_call_gamma(S, K, r, sigma, tau - e)  # gamma plus proche maturité
    dG_dTau = (g_less - g_now) / (-e)  # dérivée approx
    return -dG_dTau  # signe

def bs_call_speed_num(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # speed = dGamma/dS (approx)
    if tau <= 0.0 or sigma <= 0.0 or S <= 0.0:  # cas limites
        return 0.0  # renvoie 0
    eS = _eps_S(S)  # pas spot
    g_plus = bs_call_gamma(S + eS, K, r, sigma, tau)  # gamma au spot + eps
    g_minus = bs_call_gamma(max(1e-12, S - eS), K, r, sigma, tau)  # gamma spot - eps (protège >0)
    return (g_plus - g_minus) / (2.0 * eS)  # dérivée centrée

def bs_call_zomma_num(S: float, K: float, r: float, sigma: float, tau: float) -> float:  # zomma = dGamma/dSigma (approx)
    if tau <= 0.0 or sigma <= 0.0 or S <= 0.0:  # cas limites
        return 0.0  # renvoie 0
    e = _eps_sigma(sigma)  # pas sigma
    g_plus = bs_call_gamma(S, K, r, sigma + e, tau)  # gamma sigma+e
    g_minus = bs_call_gamma(S, K, r, max(1e-12, sigma - e), tau)  # gamma sigma-e (protège >0)
    return (g_plus - g_minus) / (2.0 * e)  # dérivée centrée

# ==========================================================
# 3) Paramètres globaux (simulation + UI)
# ==========================================================
N = 800  # nombre de pas de temps

T = 1.0  # maturité (en années)
dt = T / N  # pas de temps
mu = 0.05  # drift “monde réel” pour la trajectoire spot (juste pour simuler)

sigma_real = 0.425  # volatilité “réalisée” utilisée pour simuler le spot

S0 = 100.0  # spot initial
K = 100.0  # strike
r = 0.03  # taux sans risque

SPOT_HALF_WIDTH = 50.0  # fenêtre spot autour de K pour l’affichage FIG1
DEEP_FRAC = 0.10  # seuil “deep ITM/OTM” (10%)

INTERVAL_MS = 10  # vitesse animation (ms)
STEP_SURFACE = 8  # redessiner les surfaces tous les X pas

AUTO_EXPAND = True  # agrandir automatiquement les axes si le spot sort de la fenêtre
EXPAND_EVERY = 1  # vérifier expansion tous les X pas
PAD_FRAC_Y = 0.05  # marge sur l’axe Spot
PAD_FRAC_Z = 0.10  # marge sur l’axe prix

nT = 80  # nb points temps pour la surface BS “payoff dynamique”
nS_dyn = 120  # nb points spot pour la surface BS “payoff dynamique”

TAN_WIDTH_S = 25.0  # épaisseur “tangent sheet” delta en spot
TAN_WIDTH_T = 0.05  # épaisseur “tangent sheet” theta en temps

VOL_LEVELS = 40  # nb niveaux sigma pour “surface de vol réelle”
VOL_MIN, VOL_MAX = 0.05, 0.80  # min/max sigma pour cette surface

IV_MIN, IV_MAX = 0.00, 1.00  # bornes pour la vol implicite simulée
IV_SMOOTH_STRENGTH = 0.030  # intensité des chocs IV (random walk)
IV_EPS = 1e-6  # epsilon pour éviter 0

S2_N = 70  # grille spot fig2
V2_N = 55  # grille vol fig2
SURF2_STEP = 1  # redessiner fig2 tous les X pas

FIG2_AUTO_EXPAND = True  # fig2 suit l’expansion de fig1 (sans jamais rétrécir)
FIG2_EDGE_FRAC = 0.03  # (ici pas utilisé directement dans ta version) proximité bord
FIG2_PAD_FRAC_X = 0.08  # padding X
FIG2_PAD_FRAC_Y = 0.10  # padding Y
FIG2_PAD_FRAC_Z = 0.10  # padding Z

COLOR_BS = "lightsteelblue"  # couleur surface payoff dynamique
COLOR_DELTA = "orange"  # couleur surface delta tangent
COLOR_THETA = "green"  # couleur surface theta tangent
COLOR_VOL = "mediumpurple"  # couleur surface “vol réelle”
COLOR_VOL2 = "cyan"  # couleur “bonus” si besoin (pas obligatoire)

# ==========================================================
# 4) Outils généraux (nettoyage d’objets Matplotlib)
# ==========================================================
def safe_remove(obj):  # supprime un objet Matplotlib si possible
    if obj is None:  # si déjà nul
        return None  # rien à faire
    try:
        obj.remove()  # tentative suppression
    except Exception:
        pass  # si ça échoue, on ignore (ex: déjà supprimé)
    return None  # on renvoie None pour remettre la variable propre

# ==========================================================
# 5) Générateurs : trajectoires spot + IV + paramètres smile
# ==========================================================
def generate_iv_path(rng: np.random.Generator, n_steps: int) -> np.ndarray:  # construit une IV(t) lissée
    iv = np.empty(n_steps, dtype=float)  # tableau vide
    iv[0] = rng.uniform(IV_MIN, IV_MAX)  # point de départ aléatoire

    for k in range(1, n_steps):  # boucle temps
        shock = rng.standard_normal() * IV_SMOOTH_STRENGTH  # choc gaussien
        iv[k] = iv[k - 1] + shock  # random walk

        if iv[k] < IV_MIN:  # si sous borne
            iv[k] = IV_MIN + (IV_MIN - iv[k])  # rebond
        if iv[k] > IV_MAX:  # si au-dessus
            iv[k] = IV_MAX - (iv[k] - IV_MAX)  # rebond

        iv[k] = float(np.clip(iv[k], IV_MIN, IV_MAX))  # clamp final

    w = 9  # taille fenêtre moyenne mobile
    pad = w // 2  # padding demi-fenêtre
    kernel = np.ones(w) / w  # noyau de moyenne
    iv_pad = np.pad(iv, (pad, pad), mode="edge")  # pad aux bords
    iv_sm = np.convolve(iv_pad, kernel, mode="valid")  # moyenne mobile

    iv_sm = np.clip(iv_sm, IV_MIN, IV_MAX)  # re-clamp
    iv_sm = np.maximum(iv_sm, IV_EPS)  # évite 0
    return iv_sm  # renvoie IV(t)

def generate_smooth_param_path(rng: np.random.Generator, n_steps: int, strength: float, w: int, lo: float, hi: float):  # chemin lissé générique
    x = np.empty(n_steps, dtype=float)  # tableau
    x[0] = rng.uniform(lo, hi)  # départ aléatoire
    for k in range(1, n_steps):  # random walk
        x[k] = x[k - 1] + rng.standard_normal() * strength  # pas
    pad = w // 2  # padding
    kernel = np.ones(w) / w  # moyenne mobile
    x_pad = np.pad(x, (pad, pad), mode="edge")  # pad
    x_sm = np.convolve(x_pad, kernel, mode="valid")  # lissage
    return np.clip(x_sm, lo, hi)  # clamp bornes

def generate_run():  # génère TOUTES les séries pour une “nouvelle génération”
    global dt  # on modifie dt si T change ailleurs
    rng = np.random.default_rng()  # générateur aléatoire

    t = np.linspace(0.0, T, N + 1)  # grille temps 0..T
    tau = T - t  # time-to-maturity

    Z = rng.standard_normal(N)  # gaussiennes
    inc = (mu - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * Z  # incréments GBM

    S_path = np.empty(N + 1)  # trajectoire spot
    S_path[0] = S0  # spot initial
    S_path[1:] = S0 * np.exp(np.cumsum(inc))  # spot GBM (log-normal)

    IV_path = generate_iv_path(rng, N + 1)  # IV(t) simulée

    skew_t = generate_smooth_param_path(rng, N + 1, strength=0.012, w=17, lo=-0.55, hi=0.55)  # skew(t)
    curv_t = generate_smooth_param_path(rng, N + 1, strength=0.012, w=17, lo=0.00, hi=1.10)  # curvature(t)
    wav_t  = generate_smooth_param_path(rng, N + 1, strength=0.025, w=17, lo=0.00, hi=1.00)  # waves(t)

    C_path  = np.empty(N + 1)  # prix call
    D_path  = np.empty(N + 1)  # delta
    G_path  = np.empty(N + 1)  # gamma
    TH_path = np.empty(N + 1)  # theta
    V_path  = np.empty(N + 1)  # vega

    CH_path = np.empty(N + 1)  # charm
    CO_path = np.empty(N + 1)  # color
    VA_path = np.empty(N + 1)  # vanna
    VO_path = np.empty(N + 1)  # vomma
    ZO_path = np.empty(N + 1)  # zomma
    SP_path = np.empty(N + 1)  # speed

    for k in range(N + 1):  # calcule tout à chaque pas
        Sk = float(S_path[k])  # spot
        tauk = float(tau[k])  # tau
        sigk = float(IV_path[k])  # iv

        C_path[k]  = bs_call_price(Sk, K, r, sigk, tauk)  # prix
        D_path[k]  = bs_call_delta(Sk, K, r, sigk, tauk)  # delta
        G_path[k]  = bs_call_gamma(Sk, K, r, sigk, tauk)  # gamma
        TH_path[k] = bs_call_theta(Sk, K, r, sigk, tauk)  # theta
        V_path[k]  = bs_call_vega(Sk, K, r, sigk, tauk)  # vega

        CH_path[k] = bs_call_charm_num(Sk, K, r, sigk, tauk)  # charm (num)
        CO_path[k] = bs_call_color_num(Sk, K, r, sigk, tauk)  # color (num)
        VA_path[k] = bs_call_vanna(Sk, K, r, sigk, tauk)  # vanna (ana)
        VO_path[k] = bs_call_vomma(Sk, K, r, sigk, tauk)  # vomma (ana)
        ZO_path[k] = bs_call_zomma_num(Sk, K, r, sigk, tauk)  # zomma (num)
        SP_path[k] = bs_call_speed_num(Sk, K, r, sigk, tauk)  # speed (num)

    nU = 70  # nb points pour la “nappe” delta tangent
    u = np.linspace(-TAN_WIDTH_S, TAN_WIDTH_S, nU)  # offsets spot
    X_delta = np.tile(t, (nU, 1))  # x = temps
    Y_delta = np.tile(S_path, (nU, 1)) + u[:, None]  # y = spot autour de la trajectoire
    Z_delta = np.tile(C_path, (nU, 1)) + u[:, None] * np.tile(D_path, (nU, 1))  # z = approx linéaire via delta

    nV = 60  # nb points pour la “nappe” theta tangent
    v = np.linspace(0.0, TAN_WIDTH_T, nV)  # offsets temps
    Vt = np.minimum(v[:, None], (T - t)[None, :])  # évite d’aller au-delà de la maturité
    X_theta = np.tile(t, (nV, 1)) + Vt  # x = t + petit décalage
    Y_theta = np.tile(S_path, (nV, 1))  # y = spot
    Z_theta = np.tile(C_path, (nV, 1)) + Vt * np.tile(TH_path, (nV, 1))  # z = approx linéaire via theta

    sigmas = np.linspace(VOL_MIN, VOL_MAX, VOL_LEVELS)  # différents niveaux de vol
    Z_vol = np.empty((len(sigmas), N + 1), dtype=float)  # prix pour chaque sigma
    for si, sig in enumerate(sigmas):  # boucle sigma
        for k in range(N + 1):  # boucle temps
            Z_vol[si, k] = bs_call_price(float(S_path[k]), K, r, float(sig), float(tau[k]))  # prix BS
    X_vol = np.tile(t, (len(sigmas), 1))  # x=temps
    Y_vol = np.tile(S_path, (len(sigmas), 1))  # y=spot

    return {  # on renvoie tout dans un gros dictionnaire
        "t": t, "tau": tau,  # temps et tau
        "S_path": S_path, "C_path": C_path,  # spot et prix
        "IV_path": IV_path,  # iv
        "skew_t": skew_t, "curv_t": curv_t, "wav_t": wav_t,  # paramètres smile
        "D_path": D_path, "G_path": G_path, "TH_path": TH_path, "V_path": V_path,  # greeks base
        "CH_path": CH_path, "CO_path": CO_path, "VA_path": VA_path, "VO_path": VO_path, "ZO_path": ZO_path, "SP_path": SP_path,  # greeks avancés
        "X_delta": X_delta, "Y_delta": Y_delta, "Z_delta": Z_delta,  # nappe delta tangent
        "X_theta": X_theta, "Y_theta": Y_theta, "Z_theta": Z_theta,  # nappe theta tangent
        "X_vol": X_vol, "Y_vol": Y_vol, "Z_vol": Z_vol,  # “surface de vol réelle”
    }

# ==========================================================
# 6) Surface “Payoff dynamique” (utilise IV(t))
# ==========================================================
def build_bs_surface(S_min: float, S_max: float):
    t_grid = np.linspace(0.0, T, nT)
    S_grid = np.linspace(S_min, S_max, nS_dyn)

    TT, SS = np.meshgrid(t_grid, S_grid, indexing="xy")
    TAU = T - TT

    sig_t = np.interp(t_grid, DATA["t"], DATA["IV_path"])
    sig_t = np.maximum(sig_t, IV_EPS)

    C = np.empty_like(SS, dtype=float)
    for i in range(SS.shape[0]):
        for j in range(SS.shape[1]):
            C[i, j] = bs_call_price(float(SS[i, j]), K, r, float(sig_t[j]), float(TAU[i, j]))
    return TT, SS, C


# ==========================================================
# 7) FIG2 : “nappe de PRIX” FIXE (X=Spot ou ln(S/K), Y=σ, Z=Prix)
# ==========================================================
FIG2_USE_LOG_MONEYNESS = False  # True => x = ln(S/K) ; False => x = S

def vol_field(S_grid: np.ndarray, sigma_level_grid: np.ndarray, i: int) -> np.ndarray:
    i = int(np.clip(i, 0, int(N)))
    S = np.maximum(S_grid.astype(float), 1e-12)

    base_level = float(DATA["IV_path"][i])
    base_level = max(IV_EPS, base_level)

    skew = float(DATA["skew_t"][i])
    curv = float(DATA["curv_t"][i])
    wav  = float(DATA["wav_t"][i])

    m = np.log(S / max(1e-12, float(K)))
    lvl = sigma_level_grid.astype(float)

    sig_eff = base_level + (lvl - base_level) + skew * m + curv * (m ** 2) + wav * np.sin(6.0 * m)
    sig_eff = np.clip(sig_eff, IV_EPS, float(IV_MAX) if "IV_MAX" in globals() else 2.0)
    return sig_eff

def build_price_surface_fig2(i_ref: int, S_min: float, S_max: float):
    tau_ref = float(DATA["tau"][0])
    if tau_ref <= 0.0:
        tau_ref = 1e-6

    Sg = np.linspace(S_min, S_max, S2_N)
    Vg = np.linspace(IV_EPS, 1.0, V2_N)

    SS, VV_base = np.meshgrid(Sg, Vg, indexing="xy")
    VV_eff = vol_field(SS, VV_base, i_ref)

    Price = np.empty_like(SS, dtype=float)
    for a in range(SS.shape[0]):
        for b in range(SS.shape[1]):
            Price[a, b] = bs_call_price(float(SS[a, b]), K, r, float(VV_eff[a, b]), tau_ref)

    # => IMPORTANT : axe Y = VV_base (grille régulière)
    if FIG2_USE_LOG_MONEYNESS:
        X = np.log(np.maximum(SS, 1e-12) / max(1e-12, float(K)))
    else:
        X = SS

    Y = VV_base
    return X, Y, Price

# ==========================================================
# 8) Placement fenêtres (Tk / Qt / wx)
# ==========================================================
def _screen_size_from_manager(mgr):  # récupère taille écran (selon backend)
    w = h = None  # init
    try:  # Tk
        win = mgr.window  # fenêtre
        w = win.winfo_screenwidth()  # largeur
        h = win.winfo_screenheight()  # hauteur
        return int(w), int(h)  # renvoie
    except Exception:
        pass  # sinon on continue
    try:  # Qt
        win = mgr.window  # fenêtre
        scr = win.screen()  # écran
        geo = scr.availableGeometry()  # géométrie dispo
        return int(geo.width()), int(geo.height())  # renvoie
    except Exception:
        pass  # sinon
    return None, None  # si rien

def place_figure_rect(fig_obj, x: int, y: int, w: int, h: int):  # place une fenêtre à une position et taille
    mgr = fig_obj.canvas.manager  # manager

    try:  # Tk
        win = mgr.window  # fenêtre
        win.update_idletasks()  # actualise
        win.wm_geometry(f"{w}x{h}+{x}+{y}")  # géométrie
        return  # stop
    except Exception:
        pass  # sinon

    try:  # Qt
        win = mgr.window  # fenêtre
        win.setGeometry(x, y, w, h)  # géométrie
        return  # stop
    except Exception:
        pass  # sinon

    try:  # wxPython
        win = mgr.window  # fenêtre
        win.SetSize((w, h))  # taille
        win.Move((x, y))  # position
        return  # stop
    except Exception:
        pass  # sinon

# ==========================================================
# 9) FIG1 : setup figure principale
# ==========================================================
fig = plt.figure(figsize=(12, 7))  # crée la figure
ax = fig.add_subplot(111, projection="3d")  # axe 3D
fig.subplots_adjust(left=0.18, right=0.80, bottom=0.22)  # marges (place pour panneaux)

try:
    ax.set_proj_type("ortho")  # projection orthographique (moins de perspective)
except Exception:
    pass  # si backend ne supporte pas

ax.set_xlabel("temps t")  # label x
ax.set_ylabel("Spot S")  # label y
ax.set_zlabel("Call price C(S,t)")  # label z
ax.view_init(elev=25, azim=120)  # vue caméra
ax.set_xlim(T, 0)  # temps qui va de T -> 0 (style “compte à rebours”)

S_min_base = K - SPOT_HALF_WIDTH  # borne spot min
S_max_base = K + SPOT_HALF_WIDTH  # borne spot max
ax.set_ylim(S_min_base, S_max_base)  # y-lim
ax.set_zlim(0.0, 60.0)  # z-lim
ax.invert_yaxis()  # inverse y (effet “trader view”)

line_strike, = ax.plot([0, T], [K, K], [0, 0], lw=2)  # ligne strike

handles = {"bs": None, "delta": None, "theta": None, "vol": None}  # surfaces actives ou non

show = {  # ce qu’on affiche
    "Payoff dynamique": False,
    "Surface Δ": False,
    "Surface Theta": False,
    "Spot": True,
    "Option": True,
    "Surface de vol réelle": False,
}

alpha = {  # transparences
    "Payoff dynamique": 0.50,
    "Surface Δ": 0.50,
    "Surface Theta": 0.50,
    "Surface de vol réelle": 0.50,
}

state = {  # état animation fig1
    "i": 0,  # index temps
    "finished": False,  # fini ?
    "finish_now": False,  # bouton finish ?
    "col_max": 1,  # jusqu’à quelle colonne afficher (surfaces partielles)
    "resetting": False,  # en reset
    "paused": False,  # pause
    "ymin": S_min_base,  # borne spot min
    "ymax": S_max_base,  # borne spot max
    "zmin": 0.0,  # borne prix min
    "zmax": 60.0,  # borne prix max
}

# ==========================================================
# 10) Panneaux à gauche (moneyness + greeks)
# ==========================================================
PANEL_BG = (0.10, 0.10, 0.10)  # fond sombre

def style_panel(axp):  # “habille” un panneau
    axp.set_facecolor(PANEL_BG)  # couleur fond
    axp.set_xticks([])  # pas d’axe
    axp.set_yticks([])  # pas d’axe
    for sp in axp.spines.values():  # bordures
        sp.set_visible(True)  # visibles

ax_mny = fig.add_axes([0.02, 0.86, 0.13, 0.10])  # panel moneyness
style_panel(ax_mny)  # style
mny_main = ax_mny.text(0.5, 0.70, "ATM", ha="center", va="center", fontsize=18, fontweight="bold", color="white")  # statut
mny_sub  = ax_mny.text(0.5, 0.35, "m=0.000", ha="center", va="center", fontsize=11, fontweight="bold", color="white")  # log-moneyness
mny_iv   = ax_mny.text(0.5, 0.08, "IV=0.0%", ha="center", va="center", fontsize=11, fontweight="bold", color="violet")  # IV

ax_gr = fig.add_axes([0.02, 0.62, 0.13, 0.22])  # panel greeks base
style_panel(ax_gr)  # style
txt_delta = ax_gr.text(0.10, 0.78, "Δ: 0.0000", ha="left", va="center", fontsize=14, fontweight="bold", color="dodgerblue")  # delta
txt_gamma = ax_gr.text(0.10, 0.54, "Γ: 0.0000", ha="left", va="center", fontsize=14, fontweight="bold", color="lime")  # gamma
txt_theta = ax_gr.text(0.10, 0.30, "Θ: 0.0000", ha="left", va="center", fontsize=14, fontweight="bold", color="red")  # theta
txt_vega  = ax_gr.text(0.10, 0.06, "ν: 0.0000", ha="left", va="center", fontsize=14, fontweight="bold", color="yellow")  # vega

ax_sec = fig.add_axes([0.02, 0.28, 0.13, 0.32])  # panel greeks avancés
style_panel(ax_sec)  # style
txt_charm = ax_sec.text(0.10, 0.86, "χ: 0.0000", ha="left", va="center", fontsize=13, fontweight="bold", color="cyan")  # charm
txt_color = ax_sec.text(0.10, 0.70, "κ: 0.0000", ha="left", va="center", fontsize=13, fontweight="bold", color="magenta")  # color
txt_vanna = ax_sec.text(0.10, 0.54, "ϝ: 0.0000", ha="left", va="center", fontsize=13, fontweight="bold", color="orange")  # vanna
txt_vomma = ax_sec.text(0.10, 0.38, "ϖ: 0.0000", ha="left", va="center", fontsize=13, fontweight="bold", color="violet")  # vomma
txt_zomma = ax_sec.text(0.10, 0.22, "ζ: 0.0000", ha="left", va="center", fontsize=13, fontweight="bold", color="palegreen")  # zomma
txt_speed = ax_sec.text(0.10, 0.06, "ς: 0.0000", ha="left", va="center", fontsize=13, fontweight="bold", color="salmon")  # speed

def moneyness_status(S: float, K_: float):  # calcule statut ITM/OTM/ATM
    m = log(S / K_) if (S > 0 and K_ > 0) else 0.0  # log-moneyness
    tol = max(1e-12, 0.002 * K_)  # tol autour ATM
    deep_up = log(1.0 + DEEP_FRAC)  # deep ITM
    deep_dn = log(1.0 - DEEP_FRAC)  # deep OTM

    if m >= deep_up:  # très ITM
        return "DEEP ITM", "lime", m  # texte, couleur, m
    if m <= deep_dn:  # très OTM
        return "DEEP OTM", "red", m  # texte, couleur, m
    if S > K_ + tol:  # ITM
        return "ITM", "lime", m  # texte, couleur, m
    if S < K_ - tol:  # OTM
        return "OTM", "red", m  # texte, couleur, m
    return "ATM", "white", m  # sinon ATM

# ==========================================================
# 11) Init DATA + lignes Spot/Option
# ==========================================================
DATA = generate_run()  # première génération
TT_bs, SS_bs, C_bs = build_bs_surface(state["ymin"], state["ymax"])  # surface payoff dynamique initiale

t = DATA["t"]  # raccourci temps
S_path = DATA["S_path"]  # raccourci spot
C_path = DATA["C_path"]  # raccourci prix option

line_stock,  = ax.plot([t[0]], [S_path[0]], [0.0], lw=2)  # trajectoire spot (ligne)
line_option, = ax.plot([t[0]], [S_path[0]], [C_path[0]], lw=2)  # trajectoire option (ligne)
head_stock,  = ax.plot([t[0]], [S_path[0]], [0.0], marker="o")  # point spot
head_option, = ax.plot([t[0]], [S_path[0]], [C_path[0]], marker="o")  # point option

def option_color_from_moneyness(S: float, K_: float):  # couleur option selon ITM/OTM
    if S <= 0 or K_ <= 0:  # sécurité
        return (1.0, 1.0, 1.0, 1.0)  # blanc

    m = log(S / K_)  # log-moneyness
    deep = log(1.0 + float(DEEP_FRAC))  # seuil deep
    deep = max(deep, 1e-9)  # évite 0

    x = float(np.clip(m, -deep, deep))  # clamp dans [-deep, deep]
    u = (x + deep) / (2.0 * deep)  # normalise en [0,1]

    cmap = mcolors.LinearSegmentedColormap.from_list(  # colormap custom
        "otm_atm_itm",
        [(1.0, 0.0, 0.0), (0.75, 0.75, 0.75), (0.0, 1.0, 0.0)],  # rouge -> gris -> vert
        N=256
    )
    return cmap(u)  # RGBA

def update_left_panels(i: int):  # rafraîchit les textes à gauche
    S_now = float(DATA["S_path"][i])  # spot à i
    status, col, m = moneyness_status(S_now, float(K))  # statut
    mny_main.set_text(status)  # texte
    mny_main.set_color(col)  # couleur
    mny_sub.set_text(f"m={m: .3f}")  # affiche m

    iv_now = float(DATA["IV_path"][i])  # IV à i
    mny_iv.set_text(f"IV={100.0*iv_now: .1f}%")  # IV en %

    txt_delta.set_text(f"Δ: {float(DATA['D_path'][i]): .4f}")  # delta
    txt_gamma.set_text(f"Γ: {float(DATA['G_path'][i]): .6f}")  # gamma
    txt_theta.set_text(f"Θ: {float(DATA['TH_path'][i]): .4f}")  # theta
    txt_vega.set_text( f"ν: {float(DATA['V_path'][i]): .4f}")  # vega

    txt_charm.set_text(f"χ: {float(DATA['CH_path'][i]): .4f}")  # charm
    txt_color.set_text(f"κ: {float(DATA['CO_path'][i]): .4f}")  # color
    txt_vanna.set_text(f"ϝ: {float(DATA['VA_path'][i]): .4f}")  # vanna
    txt_vomma.set_text(f"ϖ: {float(DATA['VO_path'][i]): .4f}")  # vomma
    txt_zomma.set_text(f"ζ: {float(DATA['ZO_path'][i]): .4f}")  # zomma
    txt_speed.set_text(f"ς: {float(DATA['SP_path'][i]): .6f}")  # speed

update_left_panels(0)  # initialisation panneau gauche

col0 = option_color_from_moneyness(float(S_path[0]), float(K))  # couleur initiale option
line_option.set_color(col0)  # applique à la ligne
try:
    head_option.set_color(col0)  # applique au point
    head_option.set_markerfacecolor(col0)  # remplissage point
    head_option.set_markeredgecolor(col0)  # bord point
except Exception:
    pass  # certains backends 3D peuvent refuser

# ==========================================================
# 12) Helper surface 3D (sans ombre)
# ==========================================================
def plot_surface_noshadow(ax3d, X, Y, Z, color, a):  # helper pour surfaces uniformes
    surf = ax3d.plot_surface(  # surface
        X, Y, Z,  # grilles
        color=color,  # couleur
        alpha=a,  # transparence
        shade=False,  # pas d’ombre
        linewidth=0,  # pas de grille
        edgecolor="none",  # pas d’arêtes
        antialiased=False  # plus net/perf
    )
    try:
        surf.set_depthshade(False)  # désactive depthshade si dispo
    except Exception:
        try:
            surf._depthshade = False  # fallback selon versions
        except Exception:
            pass  # ignore
    return surf  # renvoie handle

# ==========================================================
# 13) FIG1 : redraw des surfaces
# ==========================================================
def redraw_surfaces():  # redessine surfaces selon show/alpha
    col_max = state["col_max"]  # jusqu’où on affiche

    if show["Payoff dynamique"]:  # si activée
        handles["bs"] = safe_remove(handles["bs"])  # supprime ancienne
        handles["bs"] = plot_surface_noshadow(ax, TT_bs, SS_bs, C_bs, COLOR_BS, alpha["Payoff dynamique"])  # trace nouvelle
    else:
        handles["bs"] = safe_remove(handles["bs"])  # sinon supprime

    if show["Surface Δ"]:  # delta tangent
        handles["delta"] = safe_remove(handles["delta"])  # supprime
        handles["delta"] = plot_surface_noshadow(  # retrace
            ax,
            DATA["X_delta"][:, :col_max], DATA["Y_delta"][:, :col_max], DATA["Z_delta"][:, :col_max],
            COLOR_DELTA, alpha["Surface Δ"]
        )
    else:
        handles["delta"] = safe_remove(handles["delta"])  # supprime

    if show["Surface Theta"]:  # theta tangent
        handles["theta"] = safe_remove(handles["theta"])  # supprime
        handles["theta"] = plot_surface_noshadow(  # retrace
            ax,
            DATA["X_theta"][:, :col_max], DATA["Y_theta"][:, :col_max], DATA["Z_theta"][:, :col_max],
            COLOR_THETA, alpha["Surface Theta"]
        )
    else:
        handles["theta"] = safe_remove(handles["theta"])  # supprime

    if show["Surface de vol réelle"]:  # surface sigma-levels
        handles["vol"] = safe_remove(handles["vol"])  # supprime
        handles["vol"] = plot_surface_noshadow(  # retrace
            ax,
            DATA["X_vol"][:, :col_max], DATA["Y_vol"][:, :col_max], DATA["Z_vol"][:, :col_max],
            COLOR_VOL, alpha["Surface de vol réelle"]
        )
    else:
        handles["vol"] = safe_remove(handles["vol"])  # supprime

# ==========================================================
# 14) FIG1 : auto-expand (si spot/prix sortent)
# ==========================================================
def expand_axes_and_rebuild_bs_if_needed(i: int):  # agrandit les axes si nécessaire
    global TT_bs, SS_bs, C_bs  # on met à jour la surface BS aussi
    if not AUTO_EXPAND:  # si désactivé
        return  # stop

    y = DATA["S_path"][: i + 1]  # spots déjà passés
    z = DATA["C_path"][: i + 1]  # prix déjà passés

    y_min_data = min(float(np.min(y)), K)  # min spot (inclut K)
    y_max_data = max(float(np.max(y)), K)  # max spot (inclut K)
    z_max_data = float(np.max(z))  # max prix option

    y_pad = (y_max_data - y_min_data) * PAD_FRAC_Y if y_max_data > y_min_data else 1.0  # padding y
    z_pad = (z_max_data - state["zmin"]) * PAD_FRAC_Z if z_max_data > state["zmin"] else 1.0  # padding z

    y_changed = False  # flag
    new_ymin, new_ymax = state["ymin"], state["ymax"]  # init nouvelles bornes
    new_zmax = state["zmax"]  # init

    if y_min_data < state["ymin"]:  # si dépasse en bas
        new_ymin = y_min_data - y_pad  # étend
        y_changed = True  # flag
    if y_max_data > state["ymax"]:  # si dépasse en haut
        new_ymax = y_max_data + y_pad  # étend
        y_changed = True  # flag
    if z_max_data > state["zmax"]:  # si prix dépasse
        new_zmax = z_max_data + z_pad  # étend

    if y_changed:  # si on a changé y
        state["ymin"], state["ymax"] = new_ymin, new_ymax  # update state
        ax.set_ylim(state["ymin"], state["ymax"])  # update axes
        ax.invert_yaxis()  # garde même orientation
        TT_bs, SS_bs, C_bs = build_bs_surface(state["ymin"], state["ymax"])  # reconstruit surface payoff
        if show["Payoff dynamique"]:  # si visible
            handles["bs"] = safe_remove(handles["bs"])  # supprime
            handles["bs"] = plot_surface_noshadow(ax, TT_bs, SS_bs, C_bs, COLOR_BS, alpha["Payoff dynamique"])  # retrace

    if new_zmax != state["zmax"]:  # si z max change
        state["zmax"] = new_zmax  # update
        ax.set_zlim(state["zmin"], state["zmax"])  # update zlim

# ==========================================================
# 15) FIG1 : UI à droite (checkbox + sliders + boutons)
# ==========================================================
rax = fig.add_axes([0.82, 0.58, 0.17, 0.32])  # zone pour checkbuttons
check = CheckButtons(rax, list(show.keys()), list(show.values()))  # crée les cases
rax.set_title("AFFICHER / CACHER")  # titre

def on_check(label):  # callback check/uncheck
    show[label] = not show[label]  # inverse bool
    if label == "Spot":  # si spot
        line_stock.set_visible(show[label])  # show/hide ligne
        head_stock.set_visible(show[label])  # show/hide point
    if label == "Option":  # si option
        line_option.set_visible(show[label])  # show/hide ligne
        head_option.set_visible(show[label])  # show/hide point
    redraw_surfaces()  # redraw surfaces
    fig.canvas.draw_idle()  # redraw figure

check.on_clicked(on_check)  # bind callback

ax_alpha_panel = fig.add_axes([0.82, 0.33, 0.17, 0.20])  # panneau transparence
ax_alpha_panel.set_facecolor((1, 1, 1, 0.06))  # fond translucide
ax_alpha_panel.set_xticks([])  # pas d’axe
ax_alpha_panel.set_yticks([])  # pas d’axe
for sp in ax_alpha_panel.spines.values():  # bordures
    sp.set_visible(True)  # visibles
ax_alpha_panel.text(0.5, 0.93, "TRANSPARENCE", ha="center", va="center", fontsize=11, fontweight="bold", color="white")  # titre
ax_alpha_panel.set_zorder(0)  # zorder bas

ax_sl_bs    = fig.add_axes([0.865, 0.50, 0.10, 0.022])  # slider payoff
ax_sl_delta = fig.add_axes([0.865, 0.45, 0.10, 0.022])  # slider delta
ax_sl_theta = fig.add_axes([0.865, 0.40, 0.10, 0.022])  # slider theta
ax_sl_vol   = fig.add_axes([0.865, 0.35, 0.10, 0.022])  # slider vol réelle
for a in (ax_sl_bs, ax_sl_delta, ax_sl_theta, ax_sl_vol):  # zorder
    a.set_zorder(2)  # au-dessus du panel

sl_bs    = Slider(ax_sl_bs,    "Payoff", 0.0, 1.0, valinit=alpha["Payoff dynamique"],       valfmt="%.2f")  # slider payoff
sl_delta = Slider(ax_sl_delta, "Δ",      0.0, 1.0, valinit=alpha["Surface Δ"],             valfmt="%.2f")  # slider delta
sl_theta = Slider(ax_sl_theta, "Θ",      0.0, 1.0, valinit=alpha["Surface Theta"],         valfmt="%.2f")  # slider theta
sl_vol   = Slider(ax_sl_vol,   "Vol",    0.0, 1.0, valinit=alpha["Surface de vol réelle"], valfmt="%.2f")  # slider vol

def on_alpha_change(_):  # callback slider
    alpha["Payoff dynamique"] = float(sl_bs.val)  # update
    alpha["Surface Δ"] = float(sl_delta.val)  # update
    alpha["Surface Theta"] = float(sl_theta.val)  # update
    alpha["Surface de vol réelle"] = float(sl_vol.val)  # update
    redraw_surfaces()  # redraw
    fig.canvas.draw_idle()  # redraw

sl_bs.on_changed(on_alpha_change)  # bind
sl_delta.on_changed(on_alpha_change)  # bind
sl_theta.on_changed(on_alpha_change)  # bind
sl_vol.on_changed(on_alpha_change)  # bind

ax_btn_finish = fig.add_axes([0.84, 0.26, 0.13, 0.05])  # bouton finish
btn_finish = Button(ax_btn_finish, "FINISH")  # crée
btn_finish.on_clicked(lambda _e: state.__setitem__("finish_now", True))  # met finish_now

ax_btn_pause = fig.add_axes([0.84, 0.205, 0.13, 0.045])  # bouton pause
btn_pause = Button(ax_btn_pause, "PLAY / PAUSE")  # crée

ax_btn_new = fig.add_axes([0.84, 0.15, 0.13, 0.05])  # bouton new generation
btn_new = Button(ax_btn_new, "NEW GENERATION")  # crée

# ==========================================================
# 16) FIG2 : setup (Surface de PRIX FIXE)
# ==========================================================
fig2 = plt.figure(figsize=(12, 7))
ax2 = fig2.add_subplot(111, projection="3d")

fig2.patch.set_facecolor("black")
ax2.set_facecolor("black")

# --- PANE COLORS : même gris que FIG3 (par défaut Matplotlib 3D) ---
PANE = (0.35, 0.35, 0.35, 1.0)   # ajuste à 0.30/0.40 selon ton goût

for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
    try:
        ax2.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.25)
        ax2.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.25)
        ax2.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0.25)
    except Exception:
        pass

# grille 3D discrète (optionnel, mais ça fait très FIG3)
try:
    ax2.xaxis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.35)
    ax2.yaxis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.35)
    ax2.zaxis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.35)
except Exception:
    pass

# ticks / labels en blanc (comme FIG3)
ax2.tick_params(colors="white")
ax2.xaxis.label.set_color("white")
ax2.yaxis.label.set_color("white")
ax2.zaxis.label.set_color("white")
ax2.title.set_color("white")

try:
    ax2.set_proj_type("ortho")
except Exception:
    pass

ax2.view_init(elev=18, azim=-149)
try:
    ax2.set_box_aspect((1.25, 1.00, 0.70))
except Exception:
    pass

ax2.set_title("Price Surface")
ax2.set_xlabel("Spot S" if not FIG2_USE_LOG_MONEYNESS else "log-moneyness ln(S/K)")
ax2.set_ylabel("Implied Volatility σ")
ax2.set_zlabel("Option Price V (Call)")

surf2_handle = None

# bornes fig2 (on réutilise state2 mais on va surtout s'en servir pour S_min/S_max)
state2 = {
    "xmin": 50.0, "xmax": 150.0,
    "ymin": 0.0,  "ymax": 1.0,     # sigma
    "zmin": 0.0,  "zmax": 80.0,    # price
    "needs_redraw": True,
}

def rebuild_price_surface_fig2(i_ref: int = 0):
    """Construit/trace UNE fois la surface fixe (et seulement à l'init / new generation)."""
    global surf2_handle

    S_min2 = float(state2["xmin"])
    S_max2 = float(state2["xmax"])

    X2, SIG2, P2 = build_price_surface_fig2(i_ref, S_min2, S_max2)

    surf2_handle = safe_remove(surf2_handle)
    surf2_handle = ax2.plot_surface(
        X2, SIG2, P2,
        cmap="jet",
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False
    )
    try:
        surf2_handle.set_depthshade(False)
    except Exception:
        pass

    # limites axes
    ax2.set_xlim(float(np.min(X2)), float(np.max(X2)))
    ax2.set_ylim(0.0, 1.0)

    pmin = float(np.min(P2))
    pmax = float(np.max(P2))
    pad = 0.10 * max(1e-9, (pmax - pmin))
    ax2.set_zlim(max(0.0, pmin - pad), pmax + pad)

    fig2.canvas.draw_idle()

# point (S, σ, Price) qui bouge pendant l’animation
if FIG2_USE_LOG_MONEYNESS:
    x0 = float(np.log(max(1e-12, S_path[0]) / max(1e-12, float(K))))
else:
    x0 = float(S_path[0])

pt2, = ax2.plot([x0], [float(DATA["IV_path"][0])], [float(C_path[0])],
                marker="o", lw=0, color="white")
pt2.set_visible(False)
# tige noire traversante (crée l'objet UNE FOIS)
rod2, = ax2.plot(
    [x0, x0],
    [float(DATA["IV_path"][0]), float(DATA["IV_path"][0])],
    [0.0, 1.0],              # valeurs provisoires, update_point2 va corriger
    lw=1,
    color="black"
)

def update_point2(i: int):
    S_i  = float(S_path[i])
    IV_i = float(DATA["IV_path"][i])
    C_i  = float(C_path[i])

    if FIG2_USE_LOG_MONEYNESS:
        x_i = float(np.log(max(1e-12, S_i) / max(1e-12, float(K))))
    else:
        x_i = S_i

    # point
    pt2.set_data([x_i], [IV_i])
    pt2.set_3d_properties([C_i])

    # tige traversante (dépasse en bas ET en haut)
    zmin, zmax = ax2.get_zlim()
    span = (zmax - zmin)

    z_bottom = zmin - 0.8 * span   # plus long dessous
    z_top    = zmax + 0.8 * span   # plus long dessus

    rod2.set_data([x_i, x_i], [IV_i, IV_i])
    rod2.set_3d_properties([z_bottom, z_top])

    return pt2, rod2


# ==========================================================
# 17) Reset / New generation (FIG1 + FIG2 + FIG3)
# ==========================================================
def reset_template():  # remet FIG1 aux limites de base
    global TT_bs, SS_bs, C_bs  # on reconstruit surface
    state["ymin"] = K - SPOT_HALF_WIDTH  # reset ymin
    state["ymax"] = K + SPOT_HALF_WIDTH  # reset ymax
    state["zmin"], state["zmax"] = 0.0, 60.0  # reset z
    ax.set_ylim(state["ymin"], state["ymax"])  # apply
    ax.invert_yaxis()  # keep orientation
    ax.set_zlim(state["zmin"], state["zmax"])  # apply
    ax.set_xlim(T, 0)  # apply
    line_strike.set_data_3d([0, T], [K, K], [0, 0])  # strike line
    TT_bs, SS_bs, C_bs = build_bs_surface(state["ymin"], state["ymax"])  # rebuild surface

state2 = {
    "xmin": 50.0, "xmax": 150.0,   # Spot
    "ymin": IV_EPS, "ymax": 1.0,   # Vol
    "zmin": 0.0, "zmax": 60.0,     # Prix
    "needs_redraw": True,
}

def reset_fig2_bounds():
    state2["xmin"] = float(state["ymin"])
    state2["xmax"] = float(state["ymax"])

    state2["ymin"] = float(IV_EPS)
    state2["ymax"] = 1.0

    state2["zmin"] = 0.0
    state2["zmax"] = 60.0

    state2["needs_redraw"] = True


def on_new_generation(_event):  # callback “new generation”
    global DATA, TT_bs, SS_bs, C_bs, t, S_path, C_path, surf2_handle  # globals qu’on remplace
    state["resetting"] = True  # lock animation
    state["finished"] = False  # not finished
    state["finish_now"] = False  # reset finish
    state["paused"] = False  # reset pause
    state["i"] = 0  # reset index
    state["col_max"] = 1  # reset surfaces

    for k in handles:  # supprime toutes surfaces fig1
        handles[k] = safe_remove(handles[k])  # remove

    DATA = generate_run()  # nouvelle génération data
    reset_template()  # reset fig1

    t = DATA["t"]  # refresh refs
    S_path = DATA["S_path"]  # refresh refs
    C_path = DATA["C_path"]  # refresh refs

    t0 = t[0]  # temps initial
    S0v = S_path[0]  # spot initial
    C0v = C_path[0]  # prix initial

    line_stock.set_data_3d([t0], [S0v], [0.0])  # reset ligne spot
    head_stock.set_data_3d([t0], [S0v], [0.0])  # reset point spot
    line_option.set_data_3d([t0], [S0v], [C0v])  # reset ligne option
    head_option.set_data_3d([t0], [S0v], [C0v])  # reset point option

    col_opt0 = option_color_from_moneyness(float(S0v), float(K))  # recalcul couleur option
    try:
        line_option.set_color(col_opt0)  # applique
        head_option.set_color(col_opt0)  # applique
        head_option.set_markerfacecolor(col_opt0)  # applique
        head_option.set_markeredgecolor(col_opt0)  # applique
    except Exception:
        pass  # ignore

    update_left_panels(0)  # refresh panel gauche
    redraw_surfaces()  # redraw fig1 surfaces

    surf2_handle = safe_remove(surf2_handle)

    # FIG2 : on recale les bornes spot sur fig1, puis on reconstruit UNE fois la surface fixe
    state2["xmin"] = float(state["ymin"])
    state2["xmax"] = float(state["ymax"])

    rebuild_price_surface_fig2(i_ref=0)
    update_point2(0)
    fig2.canvas.draw_idle()

    try:  # refresh fig3 si existant
        redraw_iv_surface_fig3()  # redraw fig3 surface
        fig3.canvas.draw_idle()  # redraw fig3
    except Exception as e:
        print("[NEW GENERATION] FIG3 not ready / redraw error:", e)  # log

    fig.canvas.draw_idle()  # redraw fig1
    fig2.canvas.draw_idle()  # redraw fig2

    state["resetting"] = False  # unlock

btn_new.on_clicked(on_new_generation)  # bind bouton new

def on_pause(_event):  # callback pause
    state["paused"] = not state["paused"]  # toggle
    try:
        ax_btn_pause.set_facecolor((0.18, 0.18, 0.18, 1.0) if state["paused"] else (0.94, 0.94, 0.94, 1.0))  # couleur bouton
    except Exception:
        pass  # ignore
    fig.canvas.draw_idle()  # redraw
    fig2.canvas.draw_idle()  # redraw

btn_pause.on_clicked(on_pause)  # bind bouton pause

# ==========================================================
# 18) Panel paramètres (TextBox) — propre et centré (FIG1)
# ==========================================================

ax_param_panel = fig.add_axes([0.28, 0.02, 0.44, 0.10])  # crée un panneau en bas (position en % de la figure)
ax_param_panel.set_facecolor((1, 1, 1, 0.06))           # fond blanc très transparent (effet “glass”)
ax_param_panel.set_xticks([])                           # on enlève les graduations X
ax_param_panel.set_yticks([])                           # on enlève les graduations Y

for sp in ax_param_panel.spines.values():               # boucle sur les bordures du panneau
    sp.set_visible(True)                                # on affiche les bordures

ax_param_panel.text(                                     # titre du panneau
    0.5, 0.85, "PARAMETERS",                             # position + texte
    ha="center", va="center",                            # centrage horizontal/vertical
    fontsize=11, fontweight="bold", color="white"        # style du texte
)

# --- Libellés au-dessus des TextBox (lisible pour un non-dev) ---
ax_param_panel.text(0.18, 0.62, "Spot S0",    ha="center", va="center", fontsize=10, color="white")  # label S0
ax_param_panel.text(0.50, 0.62, "Strike K",   ha="center", va="center", fontsize=10, color="white")  # label K
ax_param_panel.text(0.82, 0.62, "Maturity T", ha="center", va="center", fontsize=10, color="white")  # label T

# --- Trois zones TextBox (où tu tapes les valeurs) ---
ax_tb_s0 = fig.add_axes([0.30, 0.04, 0.13, 0.045])       # zone TextBox S0
tb_s0 = TextBox(ax_tb_s0, "", initial=str(float(S0)))     # TextBox S0 (on met la valeur actuelle)

ax_tb_k = fig.add_axes([0.44, 0.04, 0.13, 0.045])        # zone TextBox K
tb_k = TextBox(ax_tb_k, "", initial=str(float(K)))        # TextBox K

ax_tb_t = fig.add_axes([0.58, 0.04, 0.10, 0.045])        # zone TextBox T (un peu plus petite)
tb_t = TextBox(ax_tb_t, "", initial=str(float(T)))        # TextBox T

# --- Petits titres “business friendly” au-dessus de chaque box ---
for ax_tb, title in [                                    # on boucle sur chaque TextBox + son titre
    (ax_tb_s0, "ORIGINAL SPOT"),
    (ax_tb_k,  "WANTED STRIKE"),
    (ax_tb_t,  "MATURITY"),
]:
    ax_tb.text(                                          # on écrit le titre au-dessus de la box
        0.5, 1.25, title,                                # position relative à la box
        transform=ax_tb.transAxes,                       # important : coordonnées “dans la box”
        ha="center", va="bottom",                        # centrage
        fontsize=9, fontweight="bold", color="black"     # style (noir lisible sur fond clair)
    )

def apply_params(_text=None):                            # fonction appelée quand tu valides un champ
    global S0, K, T, dt                                  # on modifie des variables globales du modèle

    # 1) On essaie de lire ce que l’utilisateur a tapé
    try:
        newS0 = float(tb_s0.text)                         # convertit texte -> nombre
        newK  = float(tb_k.text)                          # convertit texte -> nombre
        newT  = float(tb_t.text)                          # convertit texte -> nombre
    except ValueError:                                    # si l’utilisateur tape un truc non-numérique
        tb_s0.set_val(str(float(S0)))                     # on remet l’ancienne valeur
        tb_k.set_val(str(float(K)))                       # on remet l’ancienne valeur
        tb_t.set_val(str(float(T)))                       # on remet l’ancienne valeur
        return                                            # on sort

    # 2) On bloque les valeurs incohérentes (ex : négatives)
    if newS0 <= 0:                                        # spot doit être > 0
        newS0 = S0                                        # sinon on garde l’ancienne valeur
    if newK <= 0:                                         # strike doit être > 0
        newK = K                                          # sinon on garde l’ancienne valeur
    if newT <= 0:                                         # maturité doit être > 0
        newT = T                                          # sinon on garde l’ancienne valeur

    # 3) On applique les nouvelles valeurs
    S0 = float(newS0)                                     # mise à jour spot initial
    K  = float(newK)                                      # mise à jour strike
    T  = float(newT)                                      # mise à jour maturité
    dt = T / N                                            # on recalcule le pas de temps (cohérent avec T)

    # 4) On met à jour des éléments dépendants de T et K sur la figure 1
    ax.set_xlim(T, 0)                                     # on garde le “compte à rebours” sur l’axe du temps
    line_strike.set_data_3d([0, T], [K, K], [0, 0])        # la ligne strike doit suivre K et T

    # 5) Le plus simple : on relance une génération complète (ça remet tout propre)
    on_new_generation(None)                               # simule un clic sur "NEW GENERATION"

# --- Quand tu appuies sur Entrée dans une box : ça applique ---
tb_s0.on_submit(apply_params)                             # bind sur S0
tb_k.on_submit(apply_params)                              # bind sur K
tb_t.on_submit(apply_params)                              # bind sur T


# ==========================================================
# 19) Animation : met à jour FIG1 + FIG2 en même temps
# ==========================================================

def update(_frame):                                       # fonction appelée à chaque frame par FuncAnimation
    if state["resetting"] or state["finished"]:           # si on est en reset ou terminé
        return ()                                         # on ne fait rien
    if state["paused"]:                                   # si on est en pause
        return ()                                         # on ne fait rien

    if state["finish_now"]:                               # si l'utilisateur a cliqué FINISH
        state["i"] = N                                    # on saute direct à la fin

    i = state["i"]                                        # index actuel dans le temps
    if i > N:                                             # si on a dépassé le dernier point
        state["finished"] = True                          # on marque fini
        state["finish_now"] = False                       # on reset le flag
        return ()                                         # on stop

    state["col_max"] = i + 1                              # jusqu’où afficher les surfaces “partielles”
    update_left_panels(i)                                 # met à jour les textes à gauche

    tloc = DATA["t"]                                      # raccourci temps
    S_loc = DATA["S_path"]                                # raccourci spot
    C_loc = DATA["C_path"]                                # raccourci prix option

    # --- FIG1 : Spot (ligne + point) ---
    if show["Spot"]:                                      # si l'utilisateur veut afficher Spot
        line_stock.set_data_3d(                           # trace la trajectoire spot jusqu'à i
            tloc[:i+1], S_loc[:i+1], np.zeros(i+1)        # z=0 pour le spot
        )
        head_stock.set_data_3d([tloc[i]], [S_loc[i]], [0.0])  # met le point “tête” spot

    # --- FIG1 : Option (ligne + point) ---
    if show["Option"]:                                    # si l'utilisateur veut afficher Option
        line_option.set_data_3d(                          # trace la trajectoire de l’option jusqu'à i
            tloc[:i+1], S_loc[:i+1], C_loc[:i+1]
        )
        head_option.set_data_3d([tloc[i]], [S_loc[i]], [C_loc[i]])  # point “tête” option

        col_opt = option_color_from_moneyness(float(S_loc[i]), float(K))  # couleur selon ITM/ATM/OTM

        try:
            line_option.set_color(col_opt)                # applique couleur à la ligne option
        except Exception:
            pass                                          # certains backends 3D sont capricieux

        try:
            head_option.set_color(col_opt)                # applique couleur au point
            head_option.set_markerfacecolor(col_opt)      # remplissage point
            head_option.set_markeredgecolor(col_opt)      # bord point
        except Exception:
            pass

    # --- Surfaces FIG1 : on les rafraîchit moins souvent pour perf ---
    if (i % STEP_SURFACE) == 0 or i == N or state["finish_now"]:  # condition de rafraîchissement
        redraw_surfaces()                                 # redraw des surfaces

    # --- Auto-expand FIG1 (si le spot sort du cadre) ---
    if AUTO_EXPAND and ((i % EXPAND_EVERY) == 0 or i == N):        # condition de check
        expand_axes_and_rebuild_bs_if_needed(i)           # agrandit limites + reconstruit surface BS

    # --- FIG2 : mise à jour du point (S, C, IV) ---
    update_point2(i)
    fig2.canvas.draw_idle()# met à jour le point blanc sur la surface
                             # demande à Matplotlib de rafraîchir fig2

    state["i"] += 1                                       # on avance dans le temps
    return ()                                             # FuncAnimation s’en fiche, mais on renvoie un tuple vide

ani = FuncAnimation(                                      # création de l’animation
    fig, update,                                          # figure + fonction update
    interval=INTERVAL_MS,                                 # vitesse (ms)
    blit=False,                                           # blit False = plus robuste en 3D
    cache_frame_data=False                                # évite d’accumuler de la RAM
)

# --- Init (premier affichage propre) ---
reset_template()                                          # remet fig1 dans un état “propre”
redraw_surfaces()                                         # dessine les surfaces initiales (si cochées)
state2["xmin"] = float(state["ymin"])
state2["xmax"] = float(state["ymax"])
rebuild_price_surface_fig2(i_ref=0)
update_point2(0)                              # dessine la surface fig2 au temps 0


# ==========================================================
# 20) FIG3 (robuste) : surface IV σ(τ, ln(S/K)) + picking + overlay
# ==========================================================

# --- Réglages de la grille FIG3 ---
FIG3_LN_MIN, FIG3_LN_MAX = -0.70, 0.70                    # log-moneyness min/max affiché
FIG3_NM = 90                                              # nb de points en log-moneyness
FIG3_NTAU = 70                                            # nb de points en tau
FIG3_CMAP = "jet_r"                                       # colormap (jet inversé)
FIG3_MAX_VOL = 1.5                                        # borne max de vol affichée

# --- Création de la fenêtre FIG3 ---
fig3 = plt.figure(figsize=(12, 7))                        # nouvelle figure
ax3 = fig3.add_subplot(111, projection="3d")              # axe 3D

# --- Thème sombre FIG3 (cohérent avec FIG2) ---
fig3.patch.set_facecolor("black")                         # fond figure noir
ax3.set_facecolor("black")                                # fond axes noir
ax3.xaxis.label.set_color("white")                        # labels blancs
ax3.yaxis.label.set_color("white")
ax3.zaxis.label.set_color("white")
ax3.title.set_color("white")
ax3.tick_params(colors="white")                           # ticks blancs

try:
    ax3.set_proj_type("ortho")                             # projection orthographique si dispo
except Exception:
    pass

ax3.set_title("Implied Volatility Surface")               # titre
ax3.set_xlabel("Time to Maturity τ")                      # axe τ
ax3.set_ylabel("log-moneyness ℓ = ln(S/K)")               # axe ln(S/K)
ax3.set_zlabel("Implied Volatility σ(τ, ℓ)")              # axe sigma
ax3.view_init(elev=30, azim=-50)                          # vue caméra

# --- Panneau d’info à droite (coords du point cliqué) ---
fig3.subplots_adjust(right=0.78)                          # laisse un espace à droite pour le panneau

ax3_info = fig3.add_axes([0.80, 0.15, 0.18, 0.70])        # axe 2D “panneau info”
ax3_info.set_facecolor((0.10, 0.10, 0.10, 0.92))          # fond sombre visible
ax3_info.set_xticks([])                                   # pas d’axe
ax3_info.set_yticks([])                                   # pas d’axe
ax3_info.set_zorder(10)                                   # au-dessus du 3D

for sp in ax3_info.spines.values():                       # bordures du panneau
    sp.set_visible(True)                                   # visibles
    sp.set_color("white")                                  # bordures blanches

ax3_info.text(                                             # titre du panneau
    0.5, 0.95, "POINT PICKED",
    ha="center", va="center",
    fontsize=11, fontweight="bold", color="white"
)

_fig3_info_txt = ax3_info.text(                            # texte “dynamique” qui change au clic
    0.06, 0.85,
    "Right-click\non the surface",
    ha="left", va="top",
    fontsize=11, color="white", family="monospace"
)

def _fig3_set_info(tau_val: float, ln_val: float, sig_val: float):  # met à jour le texte du panneau
    S_val = float(K) * float(np.exp(ln_val))               # convertit ln(S/K) -> S (utile à comprendre)
    txt = (                                                # bloc de texte lisible
        f"tau  = {tau_val: .6f}\n"
        f"ln   = {ln_val: .6f}\n"
        f"sigma= {sig_val: .6f}\n"
        f"\nS = K*exp(ln)\n"
        f"S = {S_val: .6f}\n"
        f"K = {float(K): .6f}"
    )
    _fig3_info_txt.set_text(txt)                           # applique

_surf3 = None                                              # handle de la surface (pour pouvoir la supprimer/redessiner)

# --- Cache du dernier maillage (utile pour le picking) ---
FIG3_LAST = {"TAU": None, "LN": None, "SIG": None}         # stocke les grilles et la vol lissée

# --- Overlay 2D (ligne verticale + point) qui reste visible même quand tu tournes ---
from mpl_toolkits.mplot3d import proj3d                     # projection 3D->2D
from matplotlib.lines import Line2D                         # lignes 2D “overlay”

_fig3_picked = None                                        # stocke le dernier point choisi (tau, ln, sig)
_fig3_overlay_line = None                                  # la ligne overlay
_fig3_overlay_dot = None                                   # le point overlay

def _safe_remove_local(obj):                               # supprime un objet Matplotlib en sécurité
    if obj is None:
        return None
    try:
        obj.remove()
    except Exception:
        pass
    return None

def _idx_from_tau_local(tau_val: float) -> int:            # convertit tau -> index de temps i (sur DATA)
    if float(T) <= 0.0:                                    # sécurité
        return 0
    t_val = float(T) - float(tau_val)                      # car tau = T - t
    idx = int(round((t_val / float(T)) * int(N)))          # règle de 3 pour retrouver l’index
    return int(np.clip(idx, 0, int(N)))                    # borne dans [0, N]

def _build_iv_surface_tau_ln():                            # construit (TAU, LN, SIG) pour FIG3
    tau_min = max(1e-6, float(dt))                         # tau min non nul
    tau_max = max(tau_min, float(T))                       # tau max = T

    tau_grid = np.linspace(tau_min, tau_max, int(FIG3_NTAU))  # grille tau
    ln_grid  = np.linspace(float(FIG3_LN_MIN), float(FIG3_LN_MAX), int(FIG3_NM))  # grille ln

    TAU, LN = np.meshgrid(tau_grid, ln_grid, indexing="xy") # maillage 2D
    SIG = np.empty_like(TAU, dtype=float)                  # matrice sigma résultante

    Kf = float(K)                                          # K en float (plus simple)

    for j in range(TAU.shape[1]):                          # boucle sur tau (colonne)
        tau_j = float(TAU[0, j])                           # valeur tau
        idx = _idx_from_tau_local(tau_j)                   # index temps correspondant

        base_level = float(DATA["IV_path"][idx])           # niveau IV(t) à ce temps
        base_level = max(1e-6, min(FIG3_MAX_VOL, base_level))  # clamp

        for i_ln in range(TAU.shape[0]):                   # boucle sur ln (ligne)
            ln_val = float(LN[i_ln, j])                    # valeur ln
            S_val = max(1e-6, Kf * np.exp(ln_val))         # convertit ln -> S (positif)

            sig_eff = float(                                # vol “effective” smile/skew/waves
                vol_field(
                    np.array([[S_val]], dtype=float),      # S en matrice 1x1
                    np.array([[base_level]], dtype=float), # sigma base en 1x1
                    idx                                   # temps idx
                )[0, 0]
            )
            SIG[i_ln, j] = max(1e-6, min(FIG3_MAX_VOL, sig_eff))  # clamp final

    return TAU, LN, SIG                                    # renvoie les grilles

# --- Lissage FIG3 : pour que ce soit plus “pro” (moins de vagues bruitées) ---
FIG3_SMOOTH_SIGMA = 3.2                                    # intensité du lissage spatial
FIG3_TIME_SMOOTH_ALPHA = 0.22                              # lissage temporel (EMA)

def _gauss_kernel_1d(sigma: float):                        # construit un noyau gaussien 1D
    sigma = max(1e-9, float(sigma))                        # évite sigma=0
    radius = int(max(2, round(4.0 * sigma)))               # rayon du noyau
    x = np.arange(-radius, radius + 1, dtype=float)        # axe x
    k = np.exp(-0.5 * (x / sigma) ** 2)                    # formule gaussienne
    k /= np.sum(k)                                         # normalise pour somme=1
    return k                                               # renvoie le noyau

def _smooth2d_gauss(Z: np.ndarray, sigma: float) -> np.ndarray:  # lissage 2D via 2 convolutions 1D
    k = _gauss_kernel_1d(sigma)                            # noyau
    pad = len(k) // 2                                      # padding

    Zp = np.pad(Z, ((pad, pad), (0, 0)), mode="edge")      # pad vertical
    Z0 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, Zp)  # lisse colonnes

    Zp2 = np.pad(Z0, ((0, 0), (pad, pad)), mode="edge")    # pad horizontal
    Z1 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, Zp2) # lisse lignes
    return Z1                                              # renvoie lissé

def _fig3_overlay_init():                                  # crée les objets overlay si besoin
    global _fig3_overlay_line, _fig3_overlay_dot
    if _fig3_overlay_line is not None:                    # déjà créé
        return

    _fig3_overlay_line = Line2D([0, 0], [0, 0], lw=1, color="black", zorder=10, clip_on=False)  # ligne noire
    ax3.add_line(_fig3_overlay_line)                      # ajoute à l’axe 3D

    _fig3_overlay_dot = Line2D([0], [0], marker="o", markersize=5, color="black", lw=0, zorder=11, clip_on=False)  # point noir
    ax3.add_line(_fig3_overlay_dot)                       # ajoute à l’axe 3D

def _fig3_overlay_update():                                # met à jour la ligne/point overlay
    if _fig3_picked is None:                               # si aucun point choisi
        return
    _fig3_overlay_init()                                   # crée overlay si nécessaire

    tau_val, ln_val, sig_val = _fig3_picked                # récupère le point choisi
    zmin, zmax = ax3.get_zlim()                             # limites Z actuelles
    span = zmax - zmin                                      # amplitude

    zmin_ext = zmin - 2.0 * span                            # on étend “en dessous”
    zmax_ext = zmax + 2.0 * span                            # on étend “au dessus”

    # projette 3D -> 2D pour dessiner une ligne verticale “toujours visible”
    x1, y1, _ = proj3d.proj_transform(tau_val, ln_val, float(zmin_ext), ax3.get_proj())
    x2, y2, _ = proj3d.proj_transform(tau_val, ln_val, float(zmax_ext), ax3.get_proj())
    xm, ym, _ = proj3d.proj_transform(tau_val, ln_val, float(sig_val),  ax3.get_proj())

    _fig3_overlay_line.set_data([x1, x2], [y1, y2])         # met à jour la ligne 2D
    _fig3_overlay_dot.set_data([xm], [ym])                  # met à jour le point 2D

def redraw_iv_surface_fig3():                               # redraw complet de FIG3 (surface + cache)
    global _surf3

    TAU3, LN3, SIG3 = _build_iv_surface_tau_ln()            # calcule la surface brute

    SIG3_s = _smooth2d_gauss(SIG3, sigma=FIG3_SMOOTH_SIGMA) # lissage spatial

    a = float(FIG3_TIME_SMOOTH_ALPHA)                       # alpha EMA
    if a > 0:                                               # si on veut lisser dans le temps
        prev = DATA.get("_FIG3_SIG_PREV", None)             # récupère la précédente
        if (prev is None) or (prev.shape != SIG3_s.shape):  # si pas compatible
            DATA["_FIG3_SIG_PREV"] = SIG3_s.copy()          # initialise
        else:
            SIG3_s = (1.0 - a) * prev + a * SIG3_s          # EMA (moyenne mobile exponentielle)
            DATA["_FIG3_SIG_PREV"] = SIG3_s.copy()          # sauvegarde

    SIG3_s = np.clip(SIG3_s, 1e-6, float(FIG3_MAX_VOL))     # clamp final

    FIG3_LAST["TAU"] = TAU3                                 # cache pour picking
    FIG3_LAST["LN"]  = LN3
    FIG3_LAST["SIG"] = SIG3_s

    _surf3 = _safe_remove_local(_surf3)                     # supprime l’ancienne surface
    _surf3 = ax3.plot_surface(                              # trace la nouvelle surface
        TAU3, LN3, SIG3_s,
        cmap=FIG3_CMAP,
        linewidth=0, edgecolor="none",
        antialiased=True
    )

    ax3.set_xlim(float(np.min(TAU3)), float(np.max(TAU3)))  # xlim cohérent
    ax3.set_ylim(float(np.min(LN3)),  float(np.max(LN3)))   # ylim cohérent

    zmin = float(np.nanmin(SIG3_s))                         # z min
    zmax = float(np.nanmax(SIG3_s))                         # z max
    zspan = max(1e-9, zmax - zmin)                          # span
    ax3.set_zlim(                                           # zlim avec marge
        max(0.0, zmin - 0.10 * zspan),
        min(FIG3_MAX_VOL, zmax + 0.10 * zspan)
    )

    _fig3_overlay_update()                                  # si un point existe, on repositionne overlay
    if _fig3_picked is not None:                            # si on a déjà cliqué
        _fig3_set_info(*_fig3_picked)                       # on remet les infos

    fig3.canvas.draw_idle()                                 # redraw fig3

def _fig3_pick_nearest(event):                              # trouve le point de surface le plus proche du clic
    TAU3 = FIG3_LAST["TAU"]                                 # grille tau
    LN3  = FIG3_LAST["LN"]                                  # grille ln
    SIG3 = FIG3_LAST["SIG"]                                 # grille sigma
    if TAU3 is None or LN3 is None or SIG3 is None:         # si surface pas prête
        return None

    X = TAU3.ravel()                                        # aplati en 1D
    Y = LN3.ravel()
    Z = SIG3.ravel()

    xs, ys, _ = proj3d.proj_transform(X, Y, Z, ax3.get_proj())  # projette tous les points en 2D
    pts = ax3.transData.transform(np.column_stack([xs, ys]))    # convertit en pixels écran

    dx = pts[:, 0] - float(event.x)                         # distance x au clic
    dy = pts[:, 1] - float(event.y)                         # distance y au clic
    k = int(np.argmin(dx*dx + dy*dy))                       # index du plus proche

    return float(X[k]), float(Y[k]), float(Z[k])            # renvoie le point (tau, ln, sigma)

def _on_fig3_click(event):                                  # callback souris FIG3
    global _fig3_picked
    if event.inaxes != ax3:                                 # clic ailleurs que sur la surface
        return
    if event.button != 3:                                   # bouton 3 = clic droit
        return

    picked = _fig3_pick_nearest(event)                      # on cherche le point le plus proche
    if picked is None:
        return

    tau_val, ln_val, sig_val = picked                       # décompose
    _fig3_picked = (tau_val, ln_val, sig_val)               # sauvegarde ce point

    _fig3_overlay_update()                                  # met à jour la ligne/point overlay
    _fig3_set_info(tau_val, ln_val, sig_val)                # met à jour le panneau info

    fig3.canvas.draw_idle()                                 # redraw fig3

def _on_fig3_draw(_evt):                                    # à chaque rotation/redraw, overlay doit suivre
    _fig3_overlay_update()                                  # recalcule la projection 2D

# --- Bind événements FIG3 ---
fig3.canvas.mpl_connect("button_press_event", _on_fig3_click)  # clic droit -> picking
fig3.canvas.mpl_connect("draw_event", _on_fig3_draw)           # rotation -> overlay suit

redraw_iv_surface_fig3()                                   # premier dessin FIG3


# ==========================================================
# 21) Layout fenêtres (2x2) : FIG1 gauche, FIG2 haut droite, FIG3 bas droite
# ==========================================================

fig.canvas.draw()                                          # force création fenêtre FIG1
fig2.canvas.draw()                                         # force création fenêtre FIG2
fig3.canvas.draw()                                         # force création fenêtre FIG3

sw, sh = _screen_size_from_manager(fig.canvas.manager)     # récupère taille écran (si possible)
if sw is not None and sh is not None:                      # si on a bien la taille
    half_w = sw // 2                                       # moitié largeur
    half_h = sh // 2                                       # moitié hauteur

    place_figure_rect(fig,  0,      0,      half_w, sh)    # FIG1 à gauche, pleine hauteur
    place_figure_rect(fig2, half_w, 0,      half_w, half_h)# FIG2 en haut à droite
    place_figure_rect(fig3, half_w, half_h, half_w, half_h)# FIG3 en bas à droite
_ani_keepalive = ani
plt.show()                                                 # affiche toutes les fenêtres

