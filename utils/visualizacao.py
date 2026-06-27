import pandas as pd
import matplotlib.pyplot as plt
from configs.paths import PASTA_GRAFICOS

def plotar_feature_importance(modelo, colunas, titulo="Features relevantes"):
    _garantir_pasta()
    importancias = pd.Series(modelo.feature_importances_, index=colunas)

    fig, ax = plt.subplots(figsize=(10, 6))
    importancias.nlargest(10).sort_values().plot(kind="barh", ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Relevância")
    ax.set_ylabel("Features")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / f"{titulo.lower().replace(' ', '_').replace('—', '').strip()}.png")
    plt.show()

def _garantir_pasta():
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)