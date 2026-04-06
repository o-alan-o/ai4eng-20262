import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import PowerTransformer

def generar_caso_de_uso_preparar_datos_sesgados():
    n_rows = random.randint(15, 30)
    n_features = random.randint(2, 5)
    
    # Generar datos muy sesgados (exponenciales)
    data = np.random.exponential(scale=2.0, size=(n_rows, n_features))
    feature_cols = [f'sensor_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols)
    
    # Introducir NaNs aleatorios (simulando cortes de señal)
    mask = np.random.choice([True, False], size=df.shape, p=[0.15, 0.85])
    df[mask] = np.nan
    
    # Asegurar que la primera fila no sea NaN para que interpolate no falle al inicio
    df.iloc[0] = np.random.exponential(scale=2.0, size=n_features)
    
    target_col = 'target_variable'
    df[target_col] = np.random.randint(0, 2, size=n_rows)
    
    input_data = {'df': df.copy(), 'target_col': target_col}
    
    # ---------------------------------------------------------
    # OUTPUT ESPERADO
    # ---------------------------------------------------------
    df_expected = df.copy()
    
    # 1. Pandas interpolate
    df_expected = df_expected.interpolate()
    df_expected = df_expected.bfill() # Por si queda algún NaN suelto
    
    X_expected = df_expected.drop(columns=[target_col])
    y_expected = df_expected[target_col].to_numpy()
    
    # 2. Sklearn PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    X_trans = pt.fit_transform(X_expected)
    
    output_data = (X_trans, y_expected)
    return input_data, output_data

if __name__ == "__main__":
    print("=== Test: Datos Sesgados ===")
    inp, out = generar_caso_de_uso_preparar_datos_sesgados()
    print(f"Columnas Input: {inp['df'].columns.tolist()}")
    print(f"X shape devuelta: {out[0].shape}, y shape devuelta: {out[1].shape}\n")
