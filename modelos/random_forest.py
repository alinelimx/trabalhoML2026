from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from configs.constants import RANDOM_STATE

def treinar_random_forest(X_train, y_train):
    parametros = {
        'n_estimators': [100, 200, 300], # numero de arvores na floresta
        'max_depth': [5, 10, 20, None], # profundidade maxima de cada arvore
        'min_samples_split': [2, 5, 10], # numero minimo de amostras necessarias para dividir um no
    }

    # encontra a melhor combinacao de parametros usando validacao cruzada
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE), 
        param_grid=parametros, 
        cv=5, # 5 cenarios de validacao cruzada
        scoring="f1", # f1 score como metrica de avaliacao
        n_jobs=-1)
    
    grid.fit(X_train, y_train)
    print(f"Melhores parâmetros: {grid.best_params_}")
    print(f"Melhor F1-Score (treino): {grid.best_score_:.2%}")
    return grid.best_estimator_
