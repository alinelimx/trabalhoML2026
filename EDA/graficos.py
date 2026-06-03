import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from configs.paths import PASTA_GRAFICOS

def _garantir_pasta():
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

def plotar_histograma(acervos):
    _garantir_pasta()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(acervos["Total"], bins="fd", edgecolor="black")
    ax.set_xscale("log")
    ax.axvline(np.mean(acervos["Total"]), color="red",  linestyle="--", label=f'Média: {np.mean(acervos["Total"]):.1f}')
    ax.axvline(np.median(acervos["Total"]), color="blue", linestyle=":",  label=f'Mediana: {np.median(acervos["Total"]):.1f}')
    ax.set_xlabel("Total de Empréstimos")
    ax.set_ylabel("Número de Títulos")
    ax.set_title("Distribuição de Empréstimos por Título")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "histograma.png")
    plt.close()

def plotar_boxplots(acervos, estatisticas):
    _garantir_pasta()
    top15 = estatisticas.nlargest(15, "Soma_total")["Classificação"].tolist()
    df_top15 = acervos[acervos["Classificação"].isin(top15)].copy()
    ordem = df_top15.groupby("Classificação")["Total"].sum().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_top15, x="Classificação", y="Total", order=ordem,
                flierprops=dict(marker="o", markersize=3, alpha=0.4), ax=ax)
    ax.set_yscale("log")
    ax.set_title("Classificações por Volume Total de Empréstimos")
    ax.set_xlabel("Classificação")
    ax.set_ylabel("Empréstimos por Título")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "boxplots.png")
    plt.close()

def plotar_coeficiente_variacao(estatisticas):
    _garantir_pasta()
    top20 = estatisticas[estatisticas["N_titulos"] >= 3].nlargest(20, "Soma_total").sort_values("CV_%")
    cores = ["#4C72B0" if v <= 30 else "#DD8452" if v <= 60 else "#C44E52" for v in top20["CV_%"]]

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(top20["Classificação"].astype(str), top20["CV_%"], color=cores, height=0.95)
    ax.axvline(30, color="gray",   linestyle="--", linewidth=0.8, label="CV = 30%")
    ax.axvline(60, color="orange", linestyle="--", linewidth=0.8, label="CV = 60%")
    ax.set_title("Classificações por Volume Total de Empréstimos (mín. 3 títulos)")
    ax.set_xlabel("Coeficiente de Variação (%)")
    ax.set_ylabel("Classificação")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, top20["CV_%"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "coeficienteVariacao.png")
    plt.close()

def plotar_correlacao(acervos, estatisticas, colunas_meses):
    _garantir_pasta()
    top10 = estatisticas.nlargest(10, "Soma_total")["Classificação"].tolist()
    df_top10 = acervos[acervos["Classificação"].isin(top10)]
    corr = df_top10.groupby("Classificação")[colunas_meses].sum().T.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlação de Pearson entre as 10 classificações com maior volume de empréstimos")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "matrizCorrelacao.png")
    plt.close()

def plotar_serie_temporal(acervos, colunas_meses):
    _garantir_pasta()
    total_mes = acervos[colunas_meses].sum()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(colunas_meses, total_mes.values, color="#2878b5", linewidth=2, marker="o", markersize=4)
    ax.fill_between(colunas_meses, total_mes.values, alpha=0.15, color="#4C72B0")
    ax.set_title("Quantidade de empréstimos por mês (2022–2025)", fontsize=14, pad=15)
    ax.set_xlabel("Mês")
    ax.set_ylabel("Total de empréstimos")
    ax.set_xticks(range(len(colunas_meses)))
    ax.set_xticklabels(colunas_meses, rotation=50, fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "serietemporal.png")
    plt.close()