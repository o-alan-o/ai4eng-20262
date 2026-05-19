import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def recomendar_reposicion(df, k):
    num_cols = ['demanda_media', 'demanda_std', 'lead_time_dias', 'costo_unidad', 'stock_actual']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_tmp = df.copy()
    df_tmp['cluster'] = clusters
    cluster_demand_mean = df_tmp.groupby('cluster')['demanda_media'].mean()

    max_cluster = cluster_demand_mean.idxmax()
    min_cluster = cluster_demand_mean.idxmin()

    def asignar_z(c):
        if c == max_cluster:
            return 2.05
        elif c == min_cluster:
            return 1.28
        else:
            return 1.65

    z_values = np.array([asignar_z(c) for c in clusters])

    dm_diaria = df['demanda_media'] / 7.0
    std_diaria = df['demanda_std'] / np.sqrt(7.0)
    lt = df['lead_time_dias']

    rop = dm_diaria * lt + z_values * std_diaria * np.sqrt(lt)

    resultado = pd.DataFrame({
        'sku_id': df['sku_id'],
        'cluster': clusters,
        'z_seguridad': z_values,
        'rop': rop
    })

    resultado = resultado.sort_values('rop', ascending=False).reset_index(drop=True)

    return resultado
