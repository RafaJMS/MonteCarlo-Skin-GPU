# --- START OF FILE analysis/plot_correlation.py ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run():
    print("--- Analisando Correlação de Pearson ---")
    csv_path = "lut_data_3D_GPU_step5.csv"
    if not os.path.exists(csv_path):
        csv_path = "lut_data_3D_CPU.csv"
        
    if not os.path.exists(csv_path):
        print(f"Erro: Nenhum arquivo CSV encontrado (nem GPU, nem CPU). Rode o main_generator.py em modo 3D primeiro.")
        return

    df = pd.read_csv(csv_path)
    input_params = ['Cm_fmel', 'Ch_fblood', 'Bm']
    output_rgb = ['R_int', 'G_int', 'B_int']

    correlation_matrix = df[input_params + output_rgb].corr(method='pearson')
    input_output_correlation = correlation_matrix.loc[input_params, output_rgb]

    plt.figure(figsize=(8, 6))
    sns.heatmap(input_output_correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Sensibilidade dos Parâmetros Biológicos nos Canais RGB', fontsize=14)
    plt.xlabel('Canais RGB', fontsize=12)
    plt.ylabel('Parâmetros Físicos', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
