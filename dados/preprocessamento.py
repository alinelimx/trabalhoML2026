import numpy as np 
from configs.constants import LIMITE_ALTA_DEMANDA

def preparar_features(df, COLUNAS_MESES):
    df["Total"] = df[COLUNAS_MESES].sum(axis=1)
    df["Alta demanda"] = np.where(df["Total"] > LIMITE_ALTA_DEMANDA, 1, 0)

    features = COLUNAS_MESES + ["N_titulos", "Media", "Desvio_padrao", "Min", "Max", "Q1", "Q3", "CV_%"]
    X = df[features]
    y = df["Alta demanda"]

    return X, y
    