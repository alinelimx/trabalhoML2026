import numpy as np
import pandas as pd
from configs.paths import PASTA_EDA, ESTATISTICAS_CLASSIFICACOES

def calcular_estatisticas(acervos, COLUNAS_MESES):
    acervos["Total"] = acervos[COLUNAS_MESES].sum(axis=1) # total de emprestimos de cada titulo por mes 
    
    # agrupa os acervos por classificacao e dentro de cada classificacao opera apenas sobre a coluna Total
    classificacao = acervos.groupby("Classificação")["Total"]

    estatisticas = classificacao.agg(
        #Nome_da_coluna_nova = "função"
        N_titulos= "count", # quantos titulos existem naquela classificacao
        Soma_total="sum",   # total de emprestimos da classificação
        Media="mean",       # media de emprestimos por titulo da classificacao
        Desvio_padrao="std",
        Min="min",          # titulo menos emprestado da classificacao durante o periodo todo
        Max="max",          # titulo mais emprestado
        Q1= lambda x: x.quantile(0.25), # quartil 1 (25%)
        Q3= lambda x: x.quantile(0.75), # quartil 3 (75%)
    ).reset_index()

    estatisticas["CV_%"] = (estatisticas["Desvio_padrao"] / estatisticas["Media"].replace(0, np.nan) * 100).round(2)

    PASTA_EDA.mkdir(exist_ok=True)
    estatisticas.to_csv(ESTATISTICAS_CLASSIFICACOES, index=False)
    print(f"Salvo: {ESTATISTICAS_CLASSIFICACOES}")

    return estatisticas
