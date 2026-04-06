import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import KBinsDiscretizer

def generar_caso_de_uso_preparar_datos_espaciales():
    # Minimo de 50 filas para que strategy='kmeans' y n_bins=10 no de warnings
    n_rows = random.randint(50, 80) 
    
    df = pd.DataFrame({
        'latitud': np.random.uniform(4.0, 12.0, n_rows), 
        'longitud': np.random.uniform(-75.0, -70.0, n_rows),
        'precio_m2': np.random.uniform(1000, 5000, n_rows),
        'antiguedad': np.random.randint(0, 50, n_rows)
    })
    
    target_col = 'vendido'
    df[target_col] = np.random.randint(0, 2, size=n_rows)
    
    input_data = {'df': df.copy(), 'target_col': target_col}
    
    # ---------------------------------------------------------
    # OUTPUT ESPERADO
    # ---------------------------------------------------------
    X_expected = df.drop(columns=[target_col]).copy()
    y_expected = df[target_col].to_numpy()
    
    # Sklearn KBinsDiscretizer
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
    X_expected[['latitud', 'longitud']] = kbd.fit_transform(X_expected[['latitud', 'longitud']])
    
    output_data = (X_expected.to_numpy(), y_expected)
    return input_data, output_data

if __name__ == "__main__":
    print("=== Test: Discretización Espacial ===")
    inp, out = generar_caso_de_uso_preparar_datos_espaciales()
    print(f"Primeras filas lat/lon input:\n{inp['df'][['latitud', 'longitud']].head(2)}")
    print(f"X shape devuelta: {out[0].shape}, y shape devuelta: {out[1].shape}\n")
