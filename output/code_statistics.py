# Cargar el archivo Excel "Total.xlsx" proporcionado nuevamente y leer todas sus hojas
file_path_total_again = 'Total.xlsx'
sheets_total_again = pd.read_excel(file_path_total_again, sheet_name=None)

# Preparar un DataFrame vacío para almacenar los resultados de los tests de Wilcoxon para el nuevo intento
wilcoxon_results_total_again = []

# Ejecutar el test de Wilcoxon para cada hoja y cada par de columnas "True_X" vs "False_X"
for sheet_name, sheet_data in sheets_total_again.items():
    for variation in ['0,1', '0,15', '0,2']:
        false_col = f'False_{variation}'
        true_col = f'True_{variation}'
        # Verificar si las columnas existen en la hoja actual
        if false_col in sheet_data.columns and true_col in sheet_data.columns:
            # Eliminar filas con valores NaN para evitar errores en el test
            valid_data = sheet_data[[false_col, true_col]].dropna()
            stat, p_value = wilcoxon(valid_data[false_col], valid_data[true_col])
            wilcoxon_results_total_again.append({
                'Sheet': sheet_name,
                'Variation': variation.replace(',', '.'),
                'Statistic': stat,
                'P-Value': p_value
            })

# Convertir los resultados en un DataFrame para facilitar su visualización
df_wilcoxon_results_total_again = pd.DataFrame(wilcoxon_results_total_again)
df_wilcoxon_results_total_again
