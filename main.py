import numpy as np
import matplotlib.pyplot as plt

# ======================== КОНСТАНТЫ ========================
R = 8.31446261815324          # Дж/(моль·К)
V = 210e-6                    # объём газовой пробы, м³
T_C = 80.0
T = T_C + 273.15
P0_atm = 0.5
P0_kPa = P0_atm * 101.325
y_CO2_0 = 0.96
y_N2_0  = 0.04
P_target_atm = 0.05 * P0_atm
P_target_kPa = P_target_atm * 101.325

Tc_CO2 = 304.12  # K

# ======================== НАБОРЫ ПАРАМЕТРОВ ========================
# --- Набор 1: Rahman / Wang & LeVan (из предыдущего файла) ---
params_Rahman_5A = {
    "name": "5A (Rahman/Wang-LeVan)",
    "model": "rahman",
    "q_max": 4.55,        # ммоль/г (постоянная)
    "K0": 6.76e-8,        # кПа⁻¹
    "Q": 44040.0,         # Дж/моль
    "t0": 0.30, "alpha": 0.23, "T0": 298.15,
    "t_const": None,
}
params_Rahman_13X = {
    "name": "13X (Rahman/Wang-LeVan)",
    "model": "rahman",
    "q_max": 12.9,        # ммоль/г (эффективная, не физическая!)
    "K0": 1.02e-9,        # кПа⁻¹
    "Q": 58116.0,         # Дж/моль
    "t0": None, "alpha": None, "T0": None,
    "t_const": 0.199,
}

# --- Набор 2: Dirar & Loughlin (2013), фундаментальные параметры ---
params_DL_5A = {
    "name": "5A (Dirar-Loughlin)",
    "model": "dirar",
    "Tc": Tc_CO2,
    "KH_K0": 8.550e-7,    # г/(100 г·кПа)
    "KH_Q": 44150.0,      # Дж/моль
    # qmax(T) = a*Tr + b (г/100 г) для Tr > 0.9
    "qmax_a": -34.291, "qmax_b": 55.696,
    # t(T) = t0 + alpha*(1 - T0/T) — из предыдущего файла
    "t0": 0.30, "alpha": 0.23, "T0": 298.15,
}

params_DL_13X = {
    "name": "13X (Dirar-Loughlin)",
    "model": "dirar",
    "Tc": Tc_CO2,
    "KH_K0": 2.506e-5,    # г/(100 г·кПа)
    "KH_Q": 51340.0,      # Дж/моль
    "qmax_a": -32.866, "qmax_b": 55.078,
    # t — принимаем постоянным ~0.40 (среднее по Table 10 вблизи 350 K)
    "t_const": 0.40,
}

# ======================== ФУНКЦИИ ========================
def q_toth_rahman(p_kPa, T, params):
    """Уравнение Тота в форме Rahman."""
    if p_kPa <= 0: return 0.0
    K = params["K0"] * np.exp(params["Q"] / (R * T))
    if params["t_const"] is not None:
        t = params["t_const"]
    else:
        t = params["t0"] + params["alpha"] * (1.0 - params["T0"] / T)
    qmax = params["q_max"]  # ммоль/г
    return qmax * K * p_kPa / (1.0 + (K * p_kPa)**t)**(1.0/t)

def q_toth_dirar(p_kPa, T, params):
    """Уравнение Тота с фундаментальными параметрами Dirar & Loughlin.
    Возвращает q в ммоль/г.
    """
    if p_kPa <= 0: return 0.0
    Tr = T / params["Tc"]
    # qmax, г/100 г
    qmax_g100 = params["qmax_a"] * Tr + params["qmax_b"]
    # перевод в ммоль/г: 1 г/100 г = 10 г/кг = 10/44.01 моль/кг = 0.2272 ммоль/г
    qmax_mmol = qmax_g100 * 10.0 / 44.01
    # KH, г/(100 г·кПа)
    KH = params["KH_K0"] * np.exp(params["KH_Q"] / (R * T))
    # K_T = KH / qmax_g100 (кПа⁻¹)
    KT = KH / qmax_g100
    # t
    if "t_const" in params and params["t_const"] is not None:
        t = params["t_const"]
    else:
        t = params["t0"] + params["alpha"] * (1.0 - params["T0"] / T)
    return qmax_mmol * KT * p_kPa / (1.0 + (KT * p_kPa)**t)**(1.0/t)

def q_toth(p_kPa, T, params):
    if params["model"] == "rahman":
        return q_toth_rahman(p_kPa, T, params)
    else:
        return q_toth_dirar(p_kPa, T, params)

def calc_adsorption(T, params):
    n_total_0 = P0_kPa * V / (R * T)
    n_CO2_0 = y_CO2_0 * n_total_0
    n_N2_0  = y_N2_0  * n_total_0
    p_N2 = n_N2_0 * R * T / V
    p_CO2_target = max(P_target_kPa - p_N2, 0.0)
    n_CO2_g = p_CO2_target * V / (R * T)
    n_CO2_ads = n_CO2_0 - n_CO2_g
    q_eq = q_toth(p_CO2_target, T, params)  # ммоль/г
    m_ads = (n_CO2_ads * 1000.0 / q_eq) if q_eq > 0 else np.inf  # г
    P_total_eq = p_CO2_target + p_N2
    y_CO2_eq = p_CO2_target / P_total_eq if P_total_eq > 0 else 0.0
    y_N2_eq  = p_N2 / P_total_eq if P_total_eq > 0 else 0.0

    # случай избытка
    p_CO2_excess = 1e-6
    n_CO2_g_ex = p_CO2_excess * V / (R * T)
    n_CO2_ads_ex = n_CO2_0 - n_CO2_g_ex
    q_ex = q_toth(p_CO2_excess, T, params)
    m_ads_ex = (n_CO2_ads_ex * 1000.0 / q_ex) if q_ex > 0 else np.inf
    P_total_ex = p_CO2_excess + p_N2
    y_CO2_ex = p_CO2_excess / P_total_ex if P_total_ex > 0 else 0.0
    y_N2_ex  = p_N2 / P_total_ex if P_total_ex > 0 else 0.0

    return dict(T=T, p_CO2_eq=p_CO2_target, P_total_eq=P_total_eq,
                y_CO2_eq=y_CO2_eq, y_N2_eq=y_N2_eq,
                m_ads=m_ads, q_eq=q_eq,
                m_ads_excess=m_ads_ex, p_CO2_excess=p_CO2_excess,
                P_total_excess=P_total_ex, y_CO2_excess=y_CO2_ex,
                y_N2_excess=y_N2_ex, p_N2=p_N2)

# ======================== СРАВНЕНИЕ НАБОРОВ ========================
print("=" * 70)
print("СРАВНЕНИЕ НАБОРОВ ПАРАМЕТРОВ ПРИ T = 80 °C")
print("=" * 70)

for p5, p13 in [(params_Rahman_5A, params_Rahman_13X),
                (params_DL_5A, params_DL_13X)]:
    print(f"\n--- {p5['name']} ---")
    r5 = calc_adsorption(T, p5)
    print(f"  p_CO2,равн = {r5['p_CO2_eq']:.3f} кПа | P_общ = {r5['P_total_eq']:.3f} кПа")
    print(f"  q_равн = {r5['q_eq']:.3f} ммоль/г | m_адс = {r5['m_ads']:.3f} г")
    print(f"  y_CO2 = {r5['y_CO2_eq']:.4f}, y_N2 = {r5['y_N2_eq']:.4f}")
    print(f"  [избыток] m = {r5['m_ads_excess']:.2f} г, P = {r5['P_total_excess']:.4f} кПа, "
          f"y_CO2 = {r5['y_CO2_excess']:.2e}")

    print(f"\n--- {p13['name']} ---")
    r13 = calc_adsorption(T, p13)
    print(f"  p_CO2,равн = {r13['p_CO2_eq']:.3f} кПа | P_общ = {r13['P_total_eq']:.3f} кПа")
    print(f"  q_равн = {r13['q_eq']:.3f} ммоль/г | m_адс = {r13['m_ads']:.3f} г")
    print(f"  y_CO2 = {r13['y_CO2_eq']:.4f}, y_N2 = {r13['y_N2_eq']:.4f}")
    print(f"  [избыток] m = {r13['m_ads_excess']:.2f} г, P = {r13['P_total_excess']:.4f} кПа, "
          f"y_CO2 = {r13['y_CO2_excess']:.2e}")

# ======================== ГРАФИК m(T) ========================
plt.figure(figsize=(9, 5.5))
T_range = np.arange(250, 450, 5)

for params, style in [(params_Rahman_5A, 'o-'), (params_DL_5A, 's--'),
                      (params_Rahman_13X, '^-'), (params_DL_13X, 'v--')]:
    masses = [calc_adsorption(Ti, params)["m_ads"]*1000 for Ti in T_range]
    plt.plot(T_range-273.15, masses, style, label=params["name"], linewidth=1.8, markersize=4)

plt.xlabel("Температура, C")
plt.ylabel("Необходимая масса адсорбента, мг")
plt.title(f"Масса адсорбента для снижения давления до {P_target_atm:.3f} атм\n"
          f"(CO₂ 96 %, N₂ 4 %, V = 210 мл, P₀ = 0.5 атм)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()