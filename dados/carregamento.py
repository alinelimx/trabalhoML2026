import pandas as pd
from configs.paths import ARQUIVO_CSV, ESTATISTICAS_CLASSIFICACOES

def carregar_dados():
    df_colunas = pd.read_csv(ARQUIVO_CSV, nrows=0, sep=None, engine="python", dtype={"Classificação": str})
    COLUNAS_MESES = [coluna for coluna in df_colunas.columns
                 if len(str(coluna)) == 7 and str(coluna)[4] == "-"]

    emprestimos = pd.read_csv(ARQUIVO_CSV, sep=None, engine="python")
    acervos = emprestimos.groupby(
        ["Classificação", "Código do acervo", "Título"],
        as_index=False
    )[COLUNAS_MESES].sum()

    estatistica_descritiva = pd.read_csv(ESTATISTICAS_CLASSIFICACOES, sep=None, engine="python")
    df = pd.merge(acervos, estatistica_descritiva, how='left', on='Classificação')

    return df, COLUNAS_MESES