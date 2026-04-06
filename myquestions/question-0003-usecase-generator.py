import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer

def generar_caso_de_uso_preparar_datos_texto():
    n_rows = random.randint(15, 25)
    
    vocabulario = ["excelente", "malo", "bueno", "barato", "caro", "roto", "nuevo", "usado", "oferta", "calidad", "premium"]
    descripciones = [" ".join(random.choices(vocabulario, k=random.randint(3, 8))) for _ in range(n_rows)]
    
    df = pd.DataFrame({
        'precio': np.random.uniform(10, 100, n_rows),
        'stock': np.random.randint(0, 50, n_rows),
        'descripcion': descripciones
    })
    
    target_col = 'ventas'
    df[target_col] = np.random.randint(100, 1000, n_rows)
    
    input_data = {'df': df.copy(), 'target_col': target_col}
    
    # ---------------------------------------------------------
    # OUTPUT ESPERADO
    # ---------------------------------------------------------
    X_expected = df.drop(columns=[target_col]).copy()
    y_expected = df[target_col].to_numpy()
    
    # Sklearn TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=50)
    text_features = tfidf.fit_transform(X_expected['descripcion']).toarray()
    
    # Descartar columna texto y combinar numéricas con texto vectorizado
    X_num = X_expected.drop(columns=['descripcion']).to_numpy()
    X_combined = np.hstack((X_num, text_features))
    
    output_data = (X_combined, y_expected)
    return input_data, output_data

if __name__ == "__main__":
    print("=== Test: NLP y Concatenación ===")
    inp, out = generar_caso_de_uso_preparar_datos_texto()
    print(f"Ejemplo de descripción: '{inp['df']['descripcion'].iloc[0]}'")
    print(f"X combinada shape: {out[0].shape}, y shape: {out[1].shape}\n")
