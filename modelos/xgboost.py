from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from configs.constants import RANDOM_STATE
 
def treinar_xgboost(X_train, y_train):
    parametros = {
        'n_estimators': [100, 200, 300],     # numero de arvores (rounds de boosting)
        'max_depth': [3, 5, 7],              # profundidade maxima de cada arvore
        'learning_rate': [0.01, 0.05, 0.1],  # taxa de aprendizado (shrinkage)
        'subsample': [0.8, 1.0],             # fracao de amostras usadas por arvore
    }
 
    # encontra a melhor combinacao de parametros usando validacao cruzada
    grid = GridSearchCV(
        estimator=XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',  # evita warning interno do xgboost
            n_jobs=-1
        ),
        param_grid=parametros,
        cv=5,           # 5 cenarios de validacao cruzada
        scoring='f1',   # f1 score como metrica de avaliacao
        n_jobs=-1
    )
 
    grid.fit(X_train, y_train)
    print(f"Melhores parâmetros: {grid.best_params_}")
    print(f"Melhor F1-Score (treino): {grid.best_score_:.2%}")
    return grid.best_estimator_
 