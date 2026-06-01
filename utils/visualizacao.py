import pandas as pd
import matplotlib.pyplot as plt

def plotar_feature_importance(modelo, colunas, titulo="Features relevantes"):
    importancias = pd.Series(modelo.feature_importances_, index=colunas)
    importancias.nlargest(10).sort_values().plot(kind="barh")
    plt.title(titulo)
    plt.xlabel("Relevância")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()