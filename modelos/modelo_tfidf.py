import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("Iniciando processamento do modelo de recomendação (TF-IDF)...")

    # 1. Configuração de diretórios e caminhos
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, '..', 'dados')
    caminho_resultados = os.path.join(diretorio_atual, '..', 'resultados')
    
    os.makedirs(caminho_resultados, exist_ok=True)

    # 2. Carregamento da base de conhecimento (CDD)
    try:
        with open(os.path.join(caminho_dados, 'resultado.json'), 'r', encoding='utf-8') as f:
            dicionario_cdd = json.load(f)
    except FileNotFoundError:
        print("Erro: Arquivo de dicionário CDD não encontrado.")
        return

    # 3. Carregamento e unificação dos dados de empréstimos
    arquivos_csv = ['emprestimos_2022.csv', 'emprestimos_2023.csv', 'emprestimos_2024.csv', 'emprestimos_2025.csv']
    lista_df = []
    
    for arq in arquivos_csv:
        caminho_csv = os.path.join(caminho_dados, arq)
        if os.path.exists(caminho_csv):
            lista_df.append(pd.read_csv(caminho_csv))

    if not lista_df:
        print("Erro: Base de dados de empréstimos não encontrada.")
        return

    df_global = pd.concat(lista_df, ignore_index=True)

    # 4. Pré-processamento textual (Content-Based)
    df_global['Classificação'] = df_global['Classificação'].astype(str)
    df_global['descricao_cdd'] = df_global['Classificação'].map(dicionario_cdd).fillna('Sem classificação')
    df_global['Título'] = df_global['Título'].fillna('')
    
    # Junção de features textuais
    df_global['texto_completo'] = df_global['Título'] + " " + df_global['descricao_cdd']

    # Contagem de empréstimos para critério de desempate/popularidade
    contagem = df_global['Título'].value_counts().reset_index()
    contagem.columns = ['Título', 'Total_Emprestimos']

    catalogo = df_global.drop_duplicates(subset=['Título']).copy().reset_index(drop=True)
    catalogo = pd.merge(catalogo, contagem, on='Título')

    # 5. Vetorização e cálculo de similaridade (Cosseno)
    stop_words_pt = ['o', 'a', 'os', 'as', 'de', 'do', 'da', 'em', 'para', 'com', 'que', 'ao', 'na', 'no', 'e', 'um', 'uma', 'dos', 'das']
    vetorizador = TfidfVectorizer(stop_words=stop_words_pt)
    
    matriz_tfidf = vetorizador.fit_transform(catalogo['texto_completo'])
    matriz_similaridade = cosine_similarity(matriz_tfidf)

    # 6. Geração de recomendações e extração de métricas
    K = 5
    top_livros = catalogo.sort_values(by='Total_Emprestimos', ascending=False)
    dados_exportacao = []

    print(f"Calculando similaridade e métricas para {len(top_livros)} obras cadastradas. Aguarde...")

    for index, livro_alvo in top_livros.iterrows():
        titulo_alvo = livro_alvo['Título']
        cdd_alvo = livro_alvo['Classificação']
        indice_matriz = livro_alvo.name 
        
        N_total = len(catalogo) - 1
        total_relevantes_acervo = len(catalogo[catalogo['Classificação'] == cdd_alvo]) - 1
        
        # Filtra similaridades excluindo o próprio livro alvo
        notas_texto = [(i, score) for i, score in enumerate(matriz_similaridade[indice_matriz]) if i != indice_matriz]
        notas_ordenadas = sorted(notas_texto, key=lambda x: x[1], reverse=True)[:K]
        
        acertos_tp = 0
        recomendacoes_temp = []
        
        for rank, (i, score_cosseno) in enumerate(notas_ordenadas, 1):
            livro_rec = catalogo.iloc[i]
            cdd_rec = livro_rec['Classificação']
            
            acertou = "Sim" if cdd_rec == cdd_alvo else "Não"
            if acertou == "Sim": 
                acertos_tp += 1
                
            recomendacoes_temp.append({
                'Rank_Rec': rank,
                'Livro_Recomendado': livro_rec['Título'],
                'CDD_Recomendado': cdd_rec,
                'Acerto_CDD': acertou,
                'Score_Similaridade': round(score_cosseno, 4)
            })
            
        # Cálculos de avaliação rigorosa (Gabarito CDD)
        TP = acertos_tp
        FP = K - TP
        FN = max(0, total_relevantes_acervo - TP) 
        TN = N_total - TP - FP - FN
        
        accuracy = (TP + TN) / N_total if N_total > 0 else 0
        precision = TP / K
        recall = (TP / total_relevantes_acervo) if total_relevantes_acervo > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
        # Armazenamento dos resultados
        for rec in recomendacoes_temp:
            dados_exportacao.append({
                'Livro_Alvo': titulo_alvo,
                'CDD_Alvo': cdd_alvo,
                'Rank_Rec': rec['Rank_Rec'],
                'Livro_Recomendado': rec['Livro_Recomendado'],
                'CDD_Recomendado': rec['CDD_Recomendado'],
                'Acerto_CDD': rec['Acerto_CDD'],
                'Score_Similaridade_Texto': rec['Score_Similaridade'],
                'Acuracia_Top5': accuracy,
                'Precisao_Top5': precision,
                'Recall_Top5': recall,
                'F1_Score_Top5': f1_score
            })

    # 7. Exportação de dados consolidados
    df_exportacao = pd.DataFrame(dados_exportacao)
    nome_arquivo_saida = os.path.join(caminho_resultados, 'metricas_tfidf_puro.csv')
    df_exportacao.to_csv(nome_arquivo_saida, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    
    print(f"Processamento concluído com sucesso. Resultados gerados em: {nome_arquivo_saida}")

if __name__ == "__main__":
    main()