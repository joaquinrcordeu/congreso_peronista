import inspect
import time
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

OUT_DIR = Path("tsne_learningrate_full_min40")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)
(OUT_DIR / "coords").mkdir(exist_ok=True)


# =========================================================
# CONFIGURACIÓN
# =========================================================
MIN_PALABRAS = 40

# t-SNE fijo
PERPLEXITY = 200
RANDOM_STATE = 42
EARLY_EXAGGERATION = 12
N_ITER_LIKE = 1000          # se traduce a max_iter o n_iter según tu versión
INIT = "pca"
ANGLE = 0.5

# SOLO probás esto
#LEARNING_RATES = [200, 500, 800, 1000]
# Si querés incluir auto también:
LEARNING_RATES = ["auto", 200, 500, 800, 1000]

# PCA previa
USE_PCA_PRE = True
PCA_DIM = 50

# Métrica auxiliar
TRUST_NEIGHBORS = 15
TRUST_MAX_N = 5000

# Guardado / display
SAVE_PNG = True
SHOW_PLOTS = False


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
def build_tsne(perplexity, learning_rate):
    """
    Construye TSNE compatible con distintas versiones de scikit-learn.
    """
    sig = inspect.signature(TSNE.__init__)
    valid_params = sig.parameters.keys()

    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "early_exaggeration": EARLY_EXAGGERATION,
        "learning_rate": learning_rate,
        "metric": "euclidean",   # sobre embeddings L2-normalizados
        "init": INIT,
        "verbose": 1,
        "random_state": RANDOM_STATE,
        "method": "barnes_hut",
        "angle": ANGLE,
    }

    if "max_iter" in valid_params:
        kwargs["max_iter"] = N_ITER_LIKE
    elif "n_iter" in valid_params:
        kwargs["n_iter"] = N_ITER_LIKE

    if "n_jobs" in valid_params:
        kwargs["n_jobs"] = -1

    return TSNE(**kwargs)


def safe_trustworthiness(X_high, X_low, n_neighbors=15, max_n=5000, random_state=42):
    """
    Calcula trustworthiness sobre una submuestra para no volverlo demasiado pesado.
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


def plot_projection(
    df_plot,
    x_col,
    y_col,
    title,
    save_svg_path=None,
    save_png_path=None,
    color_by_group=True,
):
    fig, ax = plt.subplots(figsize=(16, 12))

    if color_by_group:
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

    else:
        ax.scatter(
            df_plot[x_col],
            df_plot[y_col],
            s=5,
            c="#7f7f7f",
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )

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


def lr_to_str(lr):
    return str(lr).replace(".", "_")


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
# FILTRO: SOLO 40 O MÁS PALABRAS
# =========================================================
if "n_palabras" not in df.columns:
    raise ValueError("La columna 'n_palabras' no existe en el metadata.")

mask_len = df["n_palabras"] >= MIN_PALABRAS
df = df.loc[mask_len].copy().reset_index(drop=True)
emb = emb[mask_len.to_numpy()]

print(f"Filas con n_palabras >= {MIN_PALABRAS}: {len(df):,}")

if len(df) == 0:
    raise ValueError("No quedaron filas tras aplicar el filtro.")

df["grupo_plot"] = df.apply(asignar_grupo, axis=1)


# =========================================================
# NORMALIZACIÓN L2
# =========================================================
norms = np.linalg.norm(emb, axis=1, keepdims=True)
norms[norms == 0] = 1.0
emb_norm = emb / norms


# =========================================================
# PCA PREVIA
# =========================================================
if USE_PCA_PRE:
    n_pca = min(PCA_DIM, emb_norm.shape[1], len(emb_norm))
    print(f"Aplicando PCA previa a {n_pca} dimensiones...")

    pca_pre = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    X_tsne_input = pca_pre.fit_transform(emb_norm)

    pca_var_exp = float(pca_pre.explained_variance_ratio_.sum())
    print(f"Varianza explicada acumulada por PCA({n_pca}): {pca_var_exp:.4f}")
else:
    X_tsne_input = emb_norm
    n_pca = None
    pca_var_exp = np.nan
    print("Sin PCA previa.")


# =========================================================
# LOOP: SOLO LEARNING RATE
# =========================================================
results = []

for learning_rate in LEARNING_RATES:
    if PERPLEXITY >= len(X_tsne_input):
        raise ValueError(
            f"Perplexity={PERPLEXITY} debe ser menor que n_samples={len(X_tsne_input)}"
        )

    run_name = (
        f"tsne_perp{PERPLEXITY}_lr{lr_to_str(learning_rate)}"
        f"_ee{EARLY_EXAGGERATION}_min{MIN_PALABRAS}_seed{RANDOM_STATE}"
    )

    print("\n" + "=" * 100)
    print(f"CORRIENDO: {run_name}")
    print("=" * 100)

    t0 = time.time()

    tsne = build_tsne(perplexity=PERPLEXITY, learning_rate=learning_rate)
    coords = tsne.fit_transform(X_tsne_input)

    elapsed = time.time() - t0
    kl = getattr(tsne, "kl_divergence_", np.nan)

    trust = safe_trustworthiness(
        X_high=emb_norm,
        X_low=coords,
        n_neighbors=TRUST_NEIGHBORS,
        max_n=TRUST_MAX_N,
        random_state=RANDOM_STATE,
    )

    # Guardar coordenadas
    df_coords = df.copy()
    df_coords["tsne_x"] = coords[:, 0]
    df_coords["tsne_y"] = coords[:, 1]

    coords_path = OUT_DIR / "coords" / f"{run_name}.csv"
    df_coords.to_csv(coords_path, index=False)

    # Título
    title = (
        f"t-SNE | perplexity={PERPLEXITY} | learning_rate={learning_rate} | "
        f"n={len(df):,} | n_palabras>={MIN_PALABRAS}\n"
        f"KL={kl:.4f} | trustworthiness@{TRUST_NEIGHBORS}={trust:.4f}"
    )

    # Plot color
    svg_color = OUT_DIR / "plots" / f"{run_name}_color.svg"
    png_color = OUT_DIR / "plots" / f"{run_name}_color.png" if SAVE_PNG else None

    plot_projection(
        df_plot=df_coords,
        x_col="tsne_x",
        y_col="tsne_y",
        title=title,
        save_svg_path=svg_color,
        save_png_path=png_color,
        color_by_group=True,
    )

    # Plot gris
    svg_gray = OUT_DIR / "plots" / f"{run_name}_gray.svg"
    png_gray = OUT_DIR / "plots" / f"{run_name}_gray.png" if SAVE_PNG else None

    plot_projection(
        df_plot=df_coords,
        x_col="tsne_x",
        y_col="tsne_y",
        title=title,
        save_svg_path=svg_gray,
        save_png_path=png_gray,
        color_by_group=False,
    )

    # Resumen
    results.append({
        "run_name": run_name,
        "n": len(df),
        "embedding_dim": emb.shape[1],
        "min_palabras": MIN_PALABRAS,
        "perplexity": PERPLEXITY,
        "learning_rate": learning_rate,
        "early_exaggeration": EARLY_EXAGGERATION,
        "n_iter_like": N_ITER_LIKE,
        "init": INIT,
        "angle": ANGLE,
        "random_state": RANDOM_STATE,
        "use_pca_pre": USE_PCA_PRE,
        "pca_pre_dim": n_pca,
        "pca_var_explained": pca_var_exp,
        "kl_divergence": kl,
        "trustworthiness": trust,
        "runtime_sec": elapsed,
        "coords_csv": str(coords_path),
        "plot_svg_color": str(svg_color),
        "plot_svg_gray": str(svg_gray),
    })

    print(f"Tiempo: {elapsed:.1f} s")
    print(f"KL divergence: {kl:.6f}")
    print(f"Trustworthiness@{TRUST_NEIGHBORS}: {trust:.6f}")


# =========================================================
# GUARDAR RESUMEN FINAL
# =========================================================
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
                "kl_divergence",
                "trustworthiness",
                "runtime_sec",
            ]
        ]
    )
else:
    print("No hubo corridas válidas.")
