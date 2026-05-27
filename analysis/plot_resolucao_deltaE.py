import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_ciede2000
import os

print("--- Análise Perceptual: Resolução 10nm vs 5nm (CIEDE2000) ---")

csv_step10 = "lut_data_3D_GPU_step10.csv" 
csv_step5  = "lut_data_3D_GPU_step5.csv"
if not (os.path.exists(csv_step10) and os.path.exists(csv_step5)):
    print(f"Erro: Para esta análise você precisa de duas LUTs geradas. Uma com step 10 e outra com step 5.")
    print(f"Arquivos procurados: \n - {csv_step10}\n - {csv_step5}")
    exit()

# 2. Carrega os dados
df10 = pd.read_csv(csv_step10)
df5 = pd.read_csv(csv_step5)

# Garante que estão na mesma ordem
df10 = df10.sort_values(by=['idx_Bm', 'idx_Ch', 'idx_Cm']).reset_index(drop=True)
df5 = df5.sort_values(by=['idx_Bm', 'idx_Ch', 'idx_Cm']).reset_index(drop=True)

# 3. Função para calcular Delta E em lote
def calcular_delta_e_batch(df_a, df_b):
    # Pega os valores RGB, divide por 255 e remodela para o formato que a biblioteca exige
    rgb_a = df_a[['R', 'G', 'B']].values.astype(np.float64) / 255.0
    rgb_b = df_b[['R', 'G', 'B']].values.astype(np.float64) / 255.0
    
    # Redimensiona para (N, 1, 3)
    rgb_a_reshaped = rgb_a.reshape(-1, 1, 3)
    rgb_b_reshaped = rgb_b.reshape(-1, 1, 3)
    
    # Converte para LAB
    lab_a = rgb2lab(rgb_a_reshaped)
    lab_b = rgb2lab(rgb_b_reshaped)
    
    # Calcula CIEDE2000
    return deltaE_ciede2000(lab_a, lab_b).flatten()

# 4. Calcula os erros
erros_de = calcular_delta_e_batch(df10, df5)

# Adiciona ao dataframe para análise
df5['DeltaE'] = erros_de

media_de = np.mean(erros_de)
max_de = np.max(erros_de)

print(f"\nResultados da Comparação:")
print(f"Total de pontos comparados: {len(erros_de)}")
print(f"Erro Médio (ΔE): {media_de:.4f}")
print(f"Erro Máximo (ΔE): {max_de:.4f}")

if media_de < 1.0:
    print("Conclusão: Em média, a diferença entre usar Step 10 e Step 5 é IMPERCEPTÍVEL ao olho humano.")
else:
    print("Conclusão: A diferença de resolução gera um impacto perceptível na cor da pele.")

# --- 5. Geração do Gráfico de Erro ---
plt.figure(figsize=(10, 6))
plt.hist(erros_de, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
plt.axvline(x=1.0, color='red', linestyle='dashed', linewidth=2, label='Limiar de Percepção (ΔE = 1.0)')

plt.title('Distribuição do Erro Perceptual (ΔE) - Resolução Espectral 10nm vs 5nm', fontsize=14)
plt.xlabel('Erro Perceptual CIEDE2000 (ΔE)', fontsize=12)
plt.ylabel('Frequência (Quantidade de Cores)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()