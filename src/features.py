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


# --- Exercise 4.2: extração de features por janelas (5s, 50% overlap) ---

def _sliding_windows_single_label(labels, fs, win_s=5.0, overlap=0.5):
    """
    Gera janelas [start, end) com 5s e 50% overlap, mantendo APENAS janelas
    cujo rótulo (coluna 11 de 'data') é constante em toda a janela.
    Retorna: lista de (i0, i1, label)
    """
    win_n = int(round(win_s * fs))
    step = max(1, int(round(win_n * (1.0 - overlap))))
    out = []
    n = len(labels)
    for i0 in range(0, max(0, n - win_n + 1), step):
        i1 = i0 + win_n
        bloco = labels[i0:i1]
        if bloco.size < win_n:
            continue
        lab = bloco[0]
        # descartar janelas com mais do que um label
        if np.all(bloco == lab):
            out.append((i0, i1, lab))
    return out

def _iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

def _dominant_freq(x, fs):
    # FFT real; devolve frequência do pico (exclui DC)
    n = len(x)
    if n == 0:
        return np.nan
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    if freqs.size <= 1:
        return np.nan
    # potência
    P = (np.abs(X) ** 2) / n
    # ignorar DC (índice 0)
    peak_idx = np.argmax(P[1:]) + 1
    return freqs[peak_idx]

def _spectral_entropy(x, fs, eps=1e-12):
    # entropia de potência normalizada (log base e)
    n = len(x)
    if n == 0:
        return np.nan
    X = np.fft.rfft(x)
    P = (np.abs(X) ** 2)
    P = P / (np.sum(P) + eps)
    return -np.sum(P * np.log(P + eps))

def _features_1d(x, fs):
    """
    Features simples e robustas para um vetor 1D.
    Retorna tuplo (valores, nomes) num ordem estável.
    """
    if x.size == 0:
        vals = [np.nan]*9
    else:
        mean = np.mean(x)
        std = np.std(x, ddof=1) if x.size > 1 else 0.0
        med = np.median(x)
        iqr = _iqr(x)
        sma = np.mean(np.abs(x))        # Signal Magnitude Area (1D)
        energy = np.sum(x**2) / x.size  # energia média
        zc = np.sum(np.sign(x[:-1]) * np.sign(x[1:]) < 0) if x.size > 1 else 0  # zero-crossings
        df = _dominant_freq(x, fs)
        sent = _spectral_entropy(x, fs)
        vals = [mean, std, med, iqr, sma, energy, zc, df, sent]

    names = [
        "mean", "std", "median", "iqr", "sma",
        "energy", "zero_cross", "dom_freq", "spec_entropy"
    ]
    return vals, names

def _zscore_columns(X, eps=1e-12):
    X = X.astype(float, copy=True)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where(sd < eps, 1.0, sd)  # evita divisão por ~0
    X = (X - mu) / sd
    return X

def run_42_extract_features(data, acc_modules, mag_modules, gyro_modules, fs, win_s=5.0, overlap=0.5, zscore=True):
    """
    4.2 — Extrai features em janelas de 5s com 50% overlap.
    - Descarta janelas que incluem mais do que uma atividade (label na col. 11 de 'data')
    - Calcula features 1D por sinal: |Acc|, |Gyro|, |Mag|
    - Normaliza (z-score) no fim (opcional)
    Retorna: X (n_janelas x n_features), y (labels), feature_names (lista de strings)
    """
    # labels: coluna 11 do 'data' (0-based)
    labels = data[:, 11].astype(int)

    # janelas válidas (single-label)
    janelas = _sliding_windows_single_label(labels, fs, win_s, overlap)

    feats = []
    ys = []

    # nomes com prefixo por variável
    vals_dummy, base_names = _features_1d(np.array([0.0, 1.0]), fs)
    feat_names = [f"acc_{n}" for n in base_names] + \
                 [f"gyro_{n}" for n in base_names] + \
                 [f"mag_{n}" for n in base_names]

    kept, discarded = 0, 0
    for (i0, i1, lab) in janelas:
        # corta janelas por sinal
        x_acc = acc_modules[i0:i1]
        x_gyr = gyro_modules[i0:i1]
        x_mag = mag_modules[i0:i1]

        # sanity: janelas vazias em algum sinal -> descarta
        if x_acc.size == 0 or x_gyr.size == 0 or x_mag.size == 0:
            discarded += 1
            continue

        acc_f, _ = _features_1d(x_acc, fs)
        gyr_f, _ = _features_1d(x_gyr, fs)
        mag_f, _ = _features_1d(x_mag, fs)

        row = acc_f + gyr_f + mag_f
        if np.any(np.isnan(row)):
            # opcional: aceitar NaN; aqui mantemos para não perder demasiadas
            pass

        feats.append(row)
        ys.append(lab)
        kept += 1

    if len(feats) == 0:
        print("4.2: Nenhuma janela válida encontrada. Verifica fs/win_s/overlap e labels.")
        return np.empty((0, len(feat_names))), np.array([]), feat_names

    X = np.array(feats, dtype=float)
    y = np.array(ys, dtype=int)

    if zscore:
        X = _zscore_columns(X)

    print(f"4.2: janelas válidas = {kept} | descartadas (multi-label ou vazias) = {discarded}")
    print(f"4.2: X shape = {X.shape} | nº features = {X.shape[1]}")
    return X, y, feat_names

