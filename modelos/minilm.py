"""
Sistema de Recomendação de Livros com MiniLM (Sentence Transformers)

Combina três sinais para gerar recomendações:
  - Similaridade semântica entre títulos (embeddings MiniLM)
  - Similaridade hierárquica pela Classificação Decimal de Dewey (CDD)
  - Popularidade do título (total de empréstimos normalizado)

Desenvolvido originalmente no Google Colab:
https://colab.research.google.com/drive/1xIFrcIfjQytWTDvLoKMfh6fXOnPjVJ5r

Para executar no Colab, instale a dependência:
    !pip install sentence-transformers

Para executar localmente:
    pip install sentence-transformers
    python modelos/minilm.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Carregamento e pré-processamento
# ---------------------------------------------------------------------------

def carregar_dados_excel(caminho: str) -> pd.DataFrame:
    """Carrega o arquivo Excel e agrega empréstimos por título e classificação."""
    df_raw = pd.read_excel(caminho)
    df_raw["Classificação"] = df_raw["Classificação"].ffill()

    colunas_meses = [c for c in df_raw.columns if isinstance(c, str) and c[:4].isdigit()]

    df = (
        df_raw
        .groupby(["Classificação", "Título"])[colunas_meses]
        .sum()
        .reset_index()
    )

    df["total_emprestimos"] = df[colunas_meses].sum(axis=1)
    df["cdd"] = df["Classificação"].astype(str)
    df["area"] = df["cdd"].str.extract(r"(\d{3})").fillna("")
    df["Título"] = df["Título"].fillna("")

    return df


def calcular_score_popularidade(df: pd.DataFrame) -> np.ndarray:
    """Normaliza o total de empréstimos em escala log para uso como score."""
    log_emp = np.log1p(df["total_emprestimos"]).values.reshape(-1, 1)
    return MinMaxScaler().fit_transform(log_emp).flatten()


# ---------------------------------------------------------------------------
# Similaridade CDD
# ---------------------------------------------------------------------------

def sim_cdd(idx: int, cdd_list: np.ndarray) -> np.ndarray:
    """
    Calcula similaridade hierárquica entre o item `idx` e todos os outros
    com base nos dígitos da classificação CDD.

    Pontuação:
      1.0 — CDD idêntica
      0.8 — mesmo 3 primeiros dígitos (subclasse)
      0.5 — mesmos 2 primeiros dígitos (divisão)
      0.2 — mesmo 1 primeiro dígito (classe principal)
      0.0 — sem correspondência
    """
    a = cdd_list[idx]
    return np.where(
        cdd_list == a, 1.0,
        np.where(np.char.startswith(cdd_list, a[:3]), 0.8,
        np.where(np.char.startswith(cdd_list, a[:2]), 0.5,
        np.where(np.char.startswith(cdd_list, a[:1]), 0.2, 0.0)))
    )


# ---------------------------------------------------------------------------
# Busca e recomendação
# ---------------------------------------------------------------------------

def melhor_referencia(busca: str, df: pd.DataFrame):
    """Retorna o índice do título mais emprestado que contenha o termo buscado."""
    mask = df["Título"].str.upper().str.contains(busca.upper(), na=False)
    matches = df[mask]
    return matches["total_emprestimos"].idxmax() if not matches.empty else None


def recomendar(busca: str, df: pd.DataFrame, matriz_semantica: np.ndarray,
               score_popularidade: np.ndarray, top_n: int = 5):
    """Imprime as top_n recomendações para um título buscado."""
    cdd_list = df["cdd"].values
    idx = melhor_referencia(busca, df)

    if idx is None:
        print(f'Livro não encontrado: "{busca}"')
        return None

    print(f"\n{'=' * 30}\n REFERÊNCIA: {df.loc[idx, 'Título']}\n{'=' * 30}")

    scores = (
        0.30 * matriz_semantica[idx]
        + 0.30 * sim_cdd(idx, cdd_list)
        + 0.40 * score_popularidade
    )

    ranking = [i for i in np.argsort(scores)[::-1] if i != idx][:top_n]

    for i, r in enumerate(ranking, 1):
        print(f"{i}. {df.loc[r, 'Título']} (CDD {df.loc[r, 'cdd']})")

    return ranking


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def avaliar(termos: list, df: pd.DataFrame, matriz_semantica: np.ndarray,
            top_n: int = 5) -> pd.DataFrame:
    """
    Avalia o sistema de recomendação para uma lista de termos de busca.
    Usa a CDD como gabarito de relevância e calcula Precisão, Recall e F1.
    Exporta os resultados para 'metricas_minilm.xlsx'.
    """
    resultados = []

    for termo in termos:
        idx = melhor_referencia(termo, df)
        if idx is None:
            continue

        ref = df.loc[idx]
        cdd_alvo = ref["cdd"]

        ranking = [i for i in np.argsort(matriz_semantica[idx])[::-1] if i != idx][:top_n]
        relevantes = set(df[df["cdd"] == cdd_alvo].index) - {idx}

        TP = len(set(ranking) & relevantes)
        FP = top_n - TP
        FN = max(0, len(relevantes) - TP)
        TN = (len(df) - 1) - TP - FP - FN

        accuracy  = (TP + TN) / (len(df) - 1) if len(df) > 1 else 0
        precision = TP / top_n if top_n > 0 else 0
        recall    = TP / len(relevantes) if relevantes else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Exibição
        print(f"\n{'=' * 90}")
        print(f"Referência : {ref['Título']}")
        print(f"CDD        : {cdd_alvo}")
        print(f"{'=' * 90}")
        print(f"{'#':<3} {'Livro':<55} {'CDD':<10} {'Relevante'}")
        print("-" * 90)

        for pos, r in enumerate(ranking, 1):
            titulo = str(df.loc[r, "Título"])[:53]
            rel = "SIM" if r in relevantes else "NÃO"
            print(f"{pos:<3} {titulo:<55} {df.loc[r, 'cdd']:<10} {rel}")

        print(f"\nMÉTRICAS")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")

        for rank, r in enumerate(ranking, 1):
            resultados.append({
                "Livro_Alvo":               ref["Título"],
                "CDD_Alvo":                 cdd_alvo,
                "Rank_Rec":                 rank,
                "Livro_Recomendado":        df.loc[r, "Título"],
                "CDD_Recomendado":          df.loc[r, "cdd"],
                "Acerto_CDD":               "Sim" if r in relevantes else "Não",
                "Score_Similaridade_Texto": round(matriz_semantica[idx][r], 4),
                "Acuracia_Top5":            round(accuracy, 6),
                "Precisao_Top5":            round(precision, 6),
                "Recall_Top5":             round(recall, 6),
                "F1_Score_Top5":            round(f1, 6),
            })

    df_res = pd.DataFrame(resultados)

    print(f"\n{'=' * 90}\nMÉDIA GERAL\n{'=' * 90}")
    print(f"Acurácia Média : {df_res['Acuracia_Top5'].mean():.4f}")
    print(f"Precisão Média : {df_res['Precisao_Top5'].mean():.4f}")
    print(f"Recall Médio   : {df_res['Recall_Top5'].mean():.4f}")
    print(f"F1 Médio       : {df_res['F1_Score_Top5'].mean():.4f}")
    print(f"Score Médio    : {df_res['Score_Similaridade_Texto'].mean():.4f}")
    print("=" * 90)

    df_res.to_excel("metricas_minilm.xlsx", index=False)
    print("\nArquivo salvo: metricas_minilm.xlsx")

    return df_res


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Para rodar no Colab, substitua este bloco por:
    #   from google.colab import files
    #   uploaded = files.upload()
    #   caminho = list(uploaded.keys())[0]
    caminho = input("Caminho do arquivo Excel: ").strip()

    print("Carregando dados...")
    df = carregar_dados_excel(caminho)

    print("Gerando embeddings com MiniLM (pode demorar alguns minutos)...")
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = modelo.encode(df["Título"].tolist(), show_progress_bar=True)
    matriz_semantica = cosine_similarity(embeddings)
    score_popularidade = calcular_score_popularidade(df)

    termos = [
        "Cálculo - 3. ed. / 2014",
        "Fundamentos de física - 9. ed. / 2012",
        "Fundamentos de matemática elementar: 1 : conjuntos, funções - 9. ed. / 2013",
        "Lógica para computação / 2006",
        "Álgebra linear com aplicações - 10. ed. / 2012",
        "Resistência dos materiais - 7. ed. / 2010",
        "Ciência e engenharia de materiais: uma introdução - 8. ed. / 2012",
        "Fundamentos da termodinâmica / 2013",
        "Equações diferenciais - 3. ed. / 2001",
        "Termodinâmica - 7. ed. / 2013",
    ]

    df_metricas = avaliar(termos, df, matriz_semantica)
