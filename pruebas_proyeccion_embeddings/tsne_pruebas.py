import inspect
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness


# =========================================================
# RUTAS
# =========================================================
EMB_PATH = "embeddings_intervenciones_spanish_es.npy"
META_PATH = "embeddings_intervenciones_metadata.csv"

OUT_DIR = Path("tsne_pruebas")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)
(OUT_DIR / "coords").mkdir(exist_ok=True)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
FILTER_MIN_PALABRAS = 40

# Filtrado opcional por período.
# Ejemplo: [1946, 1947, 1948]
FILTER_PERIODOS = None

# Usar TODOS los discursos que cumplen el filtro de longitud.
# Como no hay submuestra, no corresponde estratificar.
SUBSAMPLE_N = None
STRATIFY_BY_GROUP = False

# PCA previa a t-SNE
PCA_PRE_TSNE_DIM = 50

# Métrica auxiliar de preservación local
TRUST_N = 5000
TRUST_NEIGHBORS = 15

# Mostrar gráficos en pantalla
SHOW_PLOTS = False

# Guardar PNG además de SVG
SAVE_PNG = True

# Semilla base
RANDOM_STATE_BASE = 42


# =========================================================
# GRID DE PARÁMETROS t-SNE
# =========================================================
PARAM_GRID = {
    "perplexity": [30, 50, 75, 100, 200, 300],
    "learning_rate": ["auto", 200, 500, 800, 1000],
    "early_exaggeration": [12],
    "n_iter_like": [1000],     # lo traduzco luego a max_iter o n_iter según versión
    "init": ["pca"],
    "angle": [0.5],
    "random_state": [42],
}


# =========================================================
# PALETA Y GRUPOS
# =========================================================
color_map = {
    "Obrero":         "#3d9a79",
    "PPF":            "#8a6bbd",
    "Político PP":    "#f5b271",
    "Político no PP": "#d48081",
}
GRUPO_ORDER = ["Político no PP", "Político PP", "Obrero", "PPF"]


def asignar_grupo(row):
    clas = row.get("clas", np.nan)
    bloque = row.get("bloque", np.nan)

    if clas == "Obrero":
        return "Obrero"
    elif clas == "PPF":
        return "PPF"
    elif clas == "Político":
        if bloque == "PP":
            return "Político PP"
        else:
            return "Político no PP"
    else:
        return np.nan


# =========================================================
# HELPERS
# =========================================================
def subsample_indices(df_in, n, stratify=True, random_state=42):
    """
    Devuelve índices de una submuestra. Si n=None o n >= len(df_in),
    devuelve todos los casos. Si stratify=True, conserva aproximadamente
    la composición por grupo_plot.
    """
    if n is None or n >= len(df_in):
        return np.arange(len(df_in))

    rng = np.random.default_rng(random_state)

    if not stratify or "grupo_plot" not in df_in.columns:
        return np.sort(rng.choice(len(df_in), size=n, replace=False))

    temp = df_in.copy()
    temp["_estrato"] = temp["grupo_plot"].fillna("Sin grupo")

    idx_final = []
    proporciones = temp["_estrato"].value_counts(normalize=True)
    estratos = proporciones.index.tolist()
    asignados = 0

    for i, estrato in enumerate(estratos):
        idx_estrato = temp.index[temp["_estrato"] == estrato].to_numpy()

        if i < len(estratos) - 1:
            k = int(round(proporciones[estrato] * n))
            k = min(k, len(idx_estrato))
        else:
            k = n - asignados
            k = min(k, len(idx_estrato))

        if k > 0:
            elegidos = rng.choice(idx_estrato, size=k, replace=False)
            idx_final.extend(elegidos.tolist())
            asignados += k

    idx_final = np.array(sorted(idx_final))

    if len(idx_final) < n:
        faltan = n - len(idx_final)
        restantes = np.setdiff1d(np.arange(len(df_in)), idx_final)
        extra = rng.choice(restantes, size=faltan, replace=False)
        idx_final = np.sort(np.concatenate([idx_final, extra]))
    elif len(idx_final) > n:
        idx_final = np.sort(rng.choice(idx_final, size=n, replace=False))

    return idx_final


def make_run_name(params):
    parts = [
        f"perp{params['perplexity']}",
        f"lr{params['learning_rate']}",
        f"ee{params['early_exaggeration']}",
        f"iter{params['n_iter_like']}",
        f"init{params['init']}",
        f"ang{str(params['angle']).replace('.', '_')}",
        f"seed{params['random_state']}",
    ]
    return "tsne_" + "_".join(parts)


def build_tsne(params):
    """
    Construye un TSNE compatible con distintas versiones de scikit-learn.
    Usa distancia euclídea sobre embeddings L2-normalizados; con eso,
    el orden por distancia euclídea equivale al orden por coseno.
    """
    sig = inspect.signature(TSNE.__init__)
    valid_params = sig.parameters.keys()

    kwargs = {
        "n_components": 2,
        "perplexity": params["perplexity"],
        "early_exaggeration": params["early_exaggeration"],
        "learning_rate": params["learning_rate"],
        "metric": "euclidean",
        "init": params["init"],
        "verbose": 1,
        "random_state": params["random_state"],
        "method": "barnes_hut",
        "angle": params["angle"],
    }

    if "max_iter" in valid_params:
        kwargs["max_iter"] = params["n_iter_like"]
    elif "n_iter" in valid_params:
        kwargs["n_iter"] = params["n_iter_like"]

    if "n_jobs" in valid_params:
        kwargs["n_jobs"] = -1

    return TSNE(**kwargs)


def plot_projection(
    df_plot,
    x_col,
    y_col,
    title,
    save_svg_path=None,
    save_png_path=None,
):
    fig, ax = plt.subplots(figsize=(16, 12))

    mask_sin = df_plot["grupo_plot"].isna()
    if mask_sin.any():
        ax.scatter(
            df_plot.loc[mask_sin, x_col],
            df_plot.loc[mask_sin, y_col],
            s=5,
            c="#d9d9d9",
            alpha=0.35,
            linewidths=0,
            zorder=1,
        )

    handles = []

    for grupo in GRUPO_ORDER:
        sub = df_plot[df_plot["grupo_plot"] == grupo]
        if sub.empty:
            continue

        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=6,
            c=color_map[grupo],
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )
        handles.append(
            Patch(facecolor=color_map[grupo], edgecolor="none", label=grupo)
        )

    if handles:
        ax.legend(handles=handles, title="Grupo", loc="best")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    plt.tight_layout()

    if save_svg_path is not None:
        plt.savefig(save_svg_path, format="svg", dpi=300, bbox_inches="tight")

    if save_png_path is not None:
        plt.savefig(save_png_path, format="png", dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def safe_trustworthiness(X_high, X_low, n_neighbors=15, max_n=5000, random_state=42):
    """
    Calcula trustworthiness sobre una submuestra si hace falta,
    para no volver la corrida demasiado lenta.
    """
    if len(X_high) > max_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_high), size=max_n, replace=False)
        Xh = X_high[idx]
        Xl = X_low[idx]
    else:
        Xh = X_high
        Xl = X_low

    return trustworthiness(Xh, Xl, n_neighbors=n_neighbors)


# =========================================================
# CARGA
# =========================================================
print("Cargando metadata...")
df = pd.read_csv(META_PATH)

print("Cargando embeddings...")
emb = np.load(EMB_PATH)

if emb.shape[0] != len(df):
    raise ValueError(
        f"Inconsistencia: embeddings={emb.shape[0]} filas, metadata={len(df)} filas."
    )

print(f"Filas cargadas: {len(df):,}")
print(f"Dimensión de embeddings: {emb.shape[1]}")


# =========================================================
# FILTROS OPCIONALES
# =========================================================
if FILTER_MIN_PALABRAS is not None:
    if "n_palabras" not in df.columns:
        raise ValueError("La columna 'n_palabras' no existe en el metadata.")
    mask_len = df["n_palabras"] >= FILTER_MIN_PALABRAS
    df = df.loc[mask_len].copy()
    emb = emb[mask_len.to_numpy()]
    print(f"Filtrado por n_palabras >= {FILTER_MIN_PALABRAS}: {len(df):,} filas")

if FILTER_PERIODOS is not None:
    if "periodo" not in df.columns:
        raise ValueError("La columna 'periodo' no existe en el metadata.")
    mask_per = df["periodo"].isin(FILTER_PERIODOS)
    df = df.loc[mask_per].copy()
    emb = emb[mask_per.to_numpy()]
    print(f"Filtrado por períodos {FILTER_PERIODOS}: {len(df):,} filas")

df = df.reset_index(drop=True)

if len(df) == 0:
    raise ValueError("No quedaron filas después de aplicar los filtros.")

df["grupo_plot"] = df.apply(asignar_grupo, axis=1)


# =========================================================
# NORMALIZACIÓN L2
# =========================================================
norms = np.linalg.norm(emb, axis=1, keepdims=True)
norms[norms == 0] = 1.0
emb_norm = emb / norms


# =========================================================
# SUBMUESTRA OPCIONAL
# =========================================================
idx_use = subsample_indices(
    df_in=df,
    n=SUBSAMPLE_N,
    stratify=STRATIFY_BY_GROUP,
    random_state=RANDOM_STATE_BASE,
)

df_use = df.iloc[idx_use].copy().reset_index(drop=True)
X_use = emb_norm[idx_use]

print(f"Filas para esta corrida: {len(df_use):,}")


# =========================================================
# PCA PREVIA A t-SNE
# =========================================================
n_pca = min(PCA_PRE_TSNE_DIM, X_use.shape[1], len(X_use))
print(f"Aplicando PCA previa a {n_pca} dimensiones...")

pca_pre = PCA(n_components=n_pca, random_state=RANDOM_STATE_BASE)
X_use_pca = pca_pre.fit_transform(X_use)

var_exp = pca_pre.explained_variance_ratio_.sum()
print(f"Varianza explicada acumulada por PCA({n_pca}): {var_exp:.4f}")


# =========================================================
# LOOP PRINCIPAL
# =========================================================
param_names = list(PARAM_GRID.keys())
param_combinations = list(product(*[PARAM_GRID[k] for k in param_names]))

print(f"Cantidad de corridas planeadas: {len(param_combinations)}")

results = []

for combo in param_combinations:
    params = dict(zip(param_names, combo))

    # t-SNE requiere perplexity < n_samples
    if params["perplexity"] >= len(X_use):
        print(
            f"Salteo corrida porque perplexity={params['perplexity']} "
            f"y n={len(X_use)}"
        )
        continue

    run_name = make_run_name(params)

    print("\n" + "=" * 100)
    print(f"CORRIENDO: {run_name}")
    print("=" * 100)

    t0 = time.time()

    tsne = build_tsne(params)
    coords = tsne.fit_transform(X_use_pca)

    elapsed = time.time() - t0

    kl = getattr(tsne, "kl_divergence_", np.nan)

    trust = safe_trustworthiness(
        X_high=X_use,
        X_low=coords,
        n_neighbors=TRUST_NEIGHBORS,
        max_n=TRUST_N,
        random_state=params["random_state"],
    )

    df_coords = df_use.copy()
    df_coords["tsne_x"] = coords[:, 0]
    df_coords["tsne_y"] = coords[:, 1]

    coords_path = OUT_DIR / "coords" / f"{run_name}.csv"
    df_coords.to_csv(coords_path, index=False)

    filtro_txt = []
    if FILTER_MIN_PALABRAS is not None:
        filtro_txt.append(f"n_palabras>={FILTER_MIN_PALABRAS}")
    if FILTER_PERIODOS is not None:
        filtro_txt.append(f"periodos={FILTER_PERIODOS}")
    filtro_txt = " | ".join(filtro_txt) if filtro_txt else "sin filtros"

    title = (
        f"t-SNE | perplexity={params['perplexity']} | "
        f"lr={params['learning_rate']} | init={params['init']} | "
        f"seed={params['random_state']}\n"
        f"KL={kl:.4f} | trustworthiness@{TRUST_NEIGHBORS}={trust:.4f} | "
        f"n={len(df_use):,} | {filtro_txt}"
    )

    svg_color = OUT_DIR / "plots" / f"{run_name}_color.svg"
    png_color = OUT_DIR / "plots" / f"{run_name}_color.png" if SAVE_PNG else None
    plot_projection(
        df_plot=df_coords,
        x_col="tsne_x",
        y_col="tsne_y",
        title=title,
        save_svg_path=svg_color,
        save_png_path=png_color,
    )

    row = {
        "run_name": run_name,
        "n_total_post_filtros": len(df),
        "n_usado": len(df_use),
        "embedding_dim": emb.shape[1],
        "pca_pre_dim": n_pca,
        "pca_var_explained": var_exp,
        "filter_min_palabras": FILTER_MIN_PALABRAS,
        "filter_periodos": None if FILTER_PERIODOS is None else ",".join(map(str, FILTER_PERIODOS)),
        "perplexity": params["perplexity"],
        "learning_rate": params["learning_rate"],
        "early_exaggeration": params["early_exaggeration"],
        "n_iter_like": params["n_iter_like"],
        "init": params["init"],
        "angle": params["angle"],
        "random_state": params["random_state"],
        "kl_divergence": kl,
        "trustworthiness": trust,
        "runtime_sec": elapsed,
        "coords_csv": str(coords_path),
        "plot_svg_color": str(svg_color),
    }
    results.append(row)

    print(f"Tiempo: {elapsed:.1f} s")
    print(f"KL divergence: {kl:.6f}")
    print(f"Trustworthiness@{TRUST_NEIGHBORS}: {trust:.6f}")

df_results = pd.DataFrame(results)

if not df_results.empty:
    df_results = df_results.sort_values(
        by=["trustworthiness", "kl_divergence"],
        ascending=[False, True]
    )

summary_path = OUT_DIR / "resumen_tsne.csv"
df_results.to_csv(summary_path, index=False)

print("\n" + "=" * 100)
print("FIN")
print(f"Resumen guardado en: {summary_path}")
print("=" * 100)

if not df_results.empty:
    print(
        df_results[
            [
                "run_name",
                "perplexity",
                "learning_rate",
                "init",
                "random_state",
                "kl_divergence",
                "trustworthiness",
                "runtime_sec",
            ]
        ].head(20)
    )
else:
    print("No hubo corridas válidas.")
