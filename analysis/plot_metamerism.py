import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run():
    print("--- Analisando Metamerismo (Ambiguidade RGB) ---")
    # 1. Descobre a pasta raiz do projeto (volta uma pasta a partir deste script)
    raiz_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Monta o caminho absoluto para o CSV na raiz do projeto
    csv_path_gpu = os.path.join(raiz_projeto, "lut_data_3D_GPU_step5.csv")
    csv_path_cpu = os.path.join(raiz_projeto, "lut_data_3D_CPU_step5.csv")
    
    if os.path.exists(csv_path_gpu):
        csv_path = csv_path_gpu
    elif os.path.exists(csv_path_cpu):
        csv_path = csv_path_cpu
    else:
        print(f"Erro: Nenhum arquivo CSV de LUT 3D encontrado na raiz ({raiz_projeto}).")
        print("Rode o main_generator.py em modo 3D primeiro.")
        return

    print(f"Lendo dados de: {csv_path}")
    df = pd.read_csv(csv_path)
    
    df['Cor_RGB'] = list(zip(df.R_int, df.G_int, df.B_int))
    contagem_cores = df['Cor_RGB'].value_counts()
    cores_metamericas = contagem_cores[contagem_cores > 1]
    frequencia_colisoes = cores_metamericas.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    cores_barras = sns.color_palette("rocket_r", len(frequencia_colisoes))
    frequencia_colisoes.plot(kind='bar', color=cores_barras, edgecolor='black')

    plt.title('Histograma de Metamerismo na Pele (Ambiguidade RGB)', fontsize=15)
    plt.xlabel('Número de Combinações Biológicas Diferentes', fontsize=12)
    plt.ylabel('Quantidade de Cores RGB', fontsize=12)

    for i, v in enumerate(frequencia_colisoes):
        plt.text(i, v + (max(frequencia_colisoes)*0.01), str(v), ha='center', fontsize=10)

    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()