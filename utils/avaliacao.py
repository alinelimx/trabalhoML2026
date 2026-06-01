from sklearn.metrics import accuracy_score, classification_report

def avaliar_modelo(modelo, X_test, y_test, nome="Modelo"):
    y_pred = modelo.predict(X_test)
    print(f"\n--- {nome} ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    return y_pred
