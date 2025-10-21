import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import f_oneway, kruskal, levene, kstest, shapiro
from scipy import stats


def per_activity_means(trans_data, values):
	activities = np.arange(1, 17)
	means = []
	for act in activities:
		mask = trans_data[:, 11] == act
		vals = values[mask]
		if vals.size == 0:
			means.append(np.nan)
		else:
			means.append(np.mean(vals))
	return activities, np.array(means)


# --- Exercise 4.1: significance of means across activities (minimal) ---

def _percent_norm_ok(resultado_norm, alpha=0.05):
    """% de grupos (atividades) com normalidade (p > alpha) no KS."""
    ps = [p for (_, p) in resultado_norm.values() if not np.isnan(p)]
    return 0.0 if not ps else 100.0 * sum(p > alpha for p in ps) / len(ps)

def choose_and_test(data, values, alpha=0.05):
    """
    Decide ANOVA (paramétrica) ou Kruskal–Wallis (não paramétrica) por variável.
    Regra simples: se >=80% dos grupos forem ~normais (KS), usa ANOVA; senão, Kruskal.
    Retorna (metodo, stat, p, pct_norm).
    """

    # Teste de normalidade por atividade (1..16)
    resultado_norm = {}
    for act in range(1, 17):
        grupo = values[data[:, 11] == act]
        if len(grupo) > 1:
            # Normaliza o grupo (z-score)
            grupo_z = (grupo - np.mean(grupo)) / np.std(grupo)
            stat, p = kstest(grupo_z, "norm")
            resultado_norm[act] = (stat, p)
        else:
            resultado_norm[act] = (np.nan, np.nan)

    pct_norm = _percent_norm_ok(resultado_norm, alpha)

    # Agrupar amostras por atividade (apenas grupos com n>=2)
    grupos = [values[data[:, 11] == act]
              for act in range(1, 17)
              if np.sum(data[:, 11] == act) > 1]

    # Regra: se >=80% normais → ANOVA; caso contrário → Kruskal
    usa_anova = pct_norm >= 80.0 and len(grupos) >= 2
    if usa_anova:
        stat, p = f_oneway(*grupos)
        metodo = "ANOVA"
    else:
        stat, p = kruskal(*grupos)
        metodo = "Kruskal–Wallis"

    return metodo, stat, p, pct_norm	

