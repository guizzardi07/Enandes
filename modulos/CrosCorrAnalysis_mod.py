
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_response_time(up_series, down_series, max_lag=80, ini=2,plot_lags=False):
    df = pd.concat([up_series, down_series], axis=1).dropna()
    df.columns = ["up", "down"]
    
    if df.empty:
        raise ValueError("No hay datos simultáneos para calcular la correlación.")

    lags = list(range(ini, max_lag + 1))
    corr_list = []

    for lag in lags:
        corr = df["up"].shift(lag).corr(df["down"])
        corr_list.append(corr)
    
    corr_arr = np.array(corr_list, dtype=float)

    # Si todas son NaN o no se pudo calcular nada
    if corr_arr.size == 0 or np.all(np.isnan(corr_arr)):
        raise ValueError("No se pudieron calcular correlaciones válidas para ningún lag.")

    # Índice del lag con mayor |correlación|
    best_idx = int(np.nanargmax(np.abs(corr_arr)))
    best_lag = lags[best_idx]

    if plot_lags:
        # Determinar lag óptimo
        abs_corrs = [abs(c) for c in corr_list]
        best_idx = abs_corrs.index(max(abs_corrs))
        best_lag = lags[best_idx]
        best_corr = corr_list[best_idx]

        # Graficar
        plt.figure(figsize=(10, 6))
        plt.plot(lags, corr_list, marker="o", label="Correlación vs lag")
        plt.axvline(best_lag, color="red", linestyle="--",
                    label=f"Lag óptimo: {best_lag} h")
        plt.scatter([best_lag], [best_corr], color="red", s=80, zorder=5)

        plt.title("Correlación entre estaciones según el lag")
        plt.xlabel("Lag (horas)")
        plt.ylabel("Correlación")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        #plt.show()
    
    return best_lag

def add_constant(X):
    return sm.add_constant(X)

