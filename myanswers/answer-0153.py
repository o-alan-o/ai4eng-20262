import pandas as pd

def top_categorias_ingresos(df, umbral_cantidad, n_top=3):
    df_filtrado = df[df['cantidad_vendida'] >= umbral_cantidad].copy()
    df_filtrado['ingreso_total'] = df_filtrado['precio'] * df_filtrado['cantidad_vendida']

    resultado = (
        df_filtrado
        .groupby('categoria')['ingreso_total']
        .sum()
        .sort_values(ascending=False)
        .head(n_top)
    )

    return resultado
