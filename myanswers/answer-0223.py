import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preparar_analisis_satelital(df_telemetria, col_momento):
    df_work = df_telemetria.copy()

    sensor_cols = [col for col in df_work.columns if col != col_momento]

    df_work[col_momento] = pd.to_datetime(df_work[col_momento], errors='coerce', format='%Y/%m/%d')

    if df_work[col_momento].notna().sum() > 0:
        median_date = df_work[col_momento].dropna().median()
    else:
        median_date = pd.Timestamp('2020-01-01')

    df_work[col_momento] = df_work[col_momento].fillna(median_date)

    df_work['dia_mision'] = df_work[col_momento].dt.day.astype(int)
    df_work['mes_operativo'] = df_work[col_momento].dt.month.astype(int)
    df_work['ciclo_semanal'] = df_work[col_momento].dt.weekday.astype(int)
    df_work['prioridad_fin_semana'] = df_work['ciclo_semanal'].isin([5, 6]).astype(int)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    semana_ohe = ohe.fit_transform(df_work[['ciclo_semanal']])

    imputer = SimpleImputer(strategy='mean')
    X_num_imputed = imputer.fit_transform(df_work[sensor_cols])

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    temporal_simple = df_work[['dia_mision', 'mes_operativo', 'prioridad_fin_semana']].to_numpy(dtype=float)
    X_final = np.hstack([X_num_scaled, temporal_simple, semana_ohe])

    return X_final
