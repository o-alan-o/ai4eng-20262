import pandas as pd
import numpy as np
import random
from sklearn.impute import KNNImputer

def generar_caso_de_uso_preparar_datos_clinicos():
    n_rows = random.randint(20, 40)
    
    niveles = ['Bajo', 'Medio', 'Alto']
    df = pd.DataFrame({
        'edad': np.random.randint(18, 80, n_rows),
        'colesterol': np.random.uniform(150, 300, n_rows),
        'nivel_riesgo': random.choices(niveles, k=n_rows)
    })
    
    # Introducir NaNs aleatorios en la columna numérica 'colesterol'
    mask = np.random.choice([True, False], size=n_rows, p=[0.2, 0.8])
    df.loc[mask, 'colesterol'] = np.nan
    
    target_col = 'costo_tratamiento'
    # Valores target exponencialmente amplios para justificar log1p
    df[target_col] = np.random.lognormal(mean=8, sigma=2, size=n_rows) 
    
    input_data = {'df': df.copy(), 'target_col': target_col}
    
    # ---------------------------------------------------------
    # OUTPUT ESPERADO
    # ---------------------------------------------------------
    df_expected = df.copy()
    
    # 1. Pandas Map para categóricas
    df_expected['nivel_riesgo'] = df_expected['nivel_riesgo'].map({'Bajo': 0, 'Medio': 1, 'Alto': 2})
    
    X_expected = df_expected.drop(columns=[target_col])
    y_expected = df_expected[target_col]
    
    # 2. Sklearn KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_expected)
    
    # 3. Numpy log1p para suavizar el target
    y_trans = np.log1p(y_expected.to_numpy())
    
    output_data = (X_imputed, y_trans)
    return input_data, output_data

if __name__ == "__main__":
    print("=== Test: Clínicos con Imputación KNN ===")
    inp, out = generar_caso_de_uso_preparar_datos_clinicos()
    print(f"Valores únicos en 'nivel_riesgo': {inp['df']['nivel_riesgo'].unique()}")
    print(f"X imputada shape: {out[0].shape}, y suavizada shape: {out[1].shape}\n")
