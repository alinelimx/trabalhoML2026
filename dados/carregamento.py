import pandas as pd
from configs.paths import ARQUIVO_CSV

def carregar_dados():
    df_colunas = pd.read_csv(ARQUIVO_CSV, nrows=0, sep=None, engine="python", dtype={"Classificação": str})
    COLUNAS_MESES = [coluna for coluna in df_colunas.columns
                 if len(str(coluna)) == 7 and str(coluna)[4] == "-"]

    emprestimos = pd.read_csv(ARQUIVO_CSV, sep=None, engine="python")
    emprestimos[COLUNAS_MESES] = emprestimos[COLUNAS_MESES].apply(pd.to_numeric, errors="coerce").fillna(0)

    acervos = emprestimos.groupby(
        ["Classificação", "Código do acervo", "Título"],
        as_index=False
    )[COLUNAS_MESES].sum()

    return acervos, COLUNAS_MESES

def juntar_acervos_com_estatisticas(acervos, estatisticas):
    return pd.merge(acervos, estatisticas, how='left', on='Classificação')