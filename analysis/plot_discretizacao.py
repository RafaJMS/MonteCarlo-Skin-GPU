# --- START OF FILE analysis/validar_discretizacao_lut.py ---
import os
import sys

# 1. PRIMEIRO: Forçamos o Python a reconhecer a pasta raiz do projeto!
# Pegamos o caminho deste arquivo, voltamos uma pasta (..) e adicionamos ao sistema.
raiz_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, raiz_projeto) # insert(0) dá prioridade máxima a essa pasta

# 2. AGORA SIM: Importamos as bibliotecas externas e o nosso código
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from skimage.color import rgb2lab, deltaE_ciede2000
import time

# Como a raiz já está no sistema, ele vai achar o main_generator perfeitamente:
from main_generator import SkinSimulation, N_PHOTONS, DEFAULT_SO2, DEFAULT_TEPI_MM

def run():
    print("--- Iniciando Validação de Erro de Discretização da LUT ---")

    # 1. Carregar os dados da LUT 3D gerada
    diretorio_atual = os.path.dirname(os.path.abspath(__file__)) # Pega o caminho da pasta 'analysis'
    
    csv_path_gpu = os.path.join(diretorio_atual, "lut_data_3D_GPU_step5.csv")
    csv_path_cpu = os.path.join(diretorio_atual, "lut_data_3D_CPU.csv")
    
    if os.path.exists(csv_path_gpu):
        csv_path = csv_path_gpu
    elif os.path.exists(csv_path_cpu):
        csv_path = csv_path_cpu
    else:
        print(f"Erro: Nenhum arquivo CSV de LUT 3D foi encontrado na pasta 'analysis'.")
        return

    print(f"Lendo dados de: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Extrair os eixos únicos para reconstruir a grade 3D
    cm_vals = np.sort(df['Cm_fmel'].unique())
    ch_vals = np.sort(df['Ch_fblood'].unique())
    bm_vals = np.sort(df['Bm'].unique())

    print(f"Eixos da LUT encontrados: Cm({len(cm_vals)}), Ch({len(ch_vals)}), Bm({len(bm_vals)})")

    # Remodelar os dados RGB para o formato da grade 3D (Bm, Ch, Cm)
    R_grid = df['R'].values.reshape((len(bm_vals), len(ch_vals), len(cm_vals)))
    G_grid = df['G'].values.reshape((len(bm_vals), len(ch_vals), len(cm_vals)))
    B_grid = df['B'].values.reshape((len(bm_vals), len(ch_vals), len(cm_vals)))

    # 3. Criar os Interpoladores Tridimensionais (Um para cada canal RGB)
    interp_R = RegularGridInterpolator((bm_vals, ch_vals, cm_vals), R_grid, method='linear')
    interp_G = RegularGridInterpolator((bm_vals, ch_vals, cm_vals), G_grid, method='linear')
    interp_B = RegularGridInterpolator((bm_vals, ch_vals, cm_vals), B_grid, method='linear')

    # 4. Gerar 100 pontos aleatórios DENTRO dos limites da LUT, mas FORA dos nós
    N_PONTOS_TESTE = 1000
    np.random.seed(42) # Seed fixa para reprodutibilidade

    random_cm = np.random.uniform(cm_vals.min(), cm_vals.max(), N_PONTOS_TESTE)
    random_ch = np.random.uniform(ch_vals.min(), ch_vals.max(), N_PONTOS_TESTE)
    random_bm = np.random.uniform(bm_vals.min(), bm_vals.max(), N_PONTOS_TESTE)

    pontos_aleatorios = np.vstack((random_bm, random_ch, random_cm)).T

    # 5. Obter as cores APROXIMADAS pela LUT via Interpolação
    cores_lut_rgb = np.zeros((N_PONTOS_TESTE, 3))
    cores_lut_rgb[:, 0] = interp_R(pontos_aleatorios)
    cores_lut_rgb[:, 1] = interp_G(pontos_aleatorios)
    cores_lut_rgb[:, 2] = interp_B(pontos_aleatorios)
    cores_lut_rgb = np.clip(cores_lut_rgb, 0, 255)

    # 6. Obter as cores EXATAS via Simulação Monte Carlo Direta (Ground Truth)
    print(f"Simulando {N_PONTOS_TESTE} pontos aleatórios no Simulador para obter o Ground Truth...")
    
    skin_sim = SkinSimulation()
    cores_gt_rgb = np.zeros((N_PONTOS_TESTE, 3))
    tepi_cm = DEFAULT_TEPI_MM / 10.0

    start_time = time.time()
    for i in range(N_PONTOS_TESTE):
        bm_val, ch_val, cm_val = pontos_aleatorios[i]
        
        # Montamos a tupla de parâmetros.
        parametros_pixel = (cm_val, ch_val, bm_val, DEFAULT_SO2, tepi_cm, N_PHOTONS, 0, 0, 0)
        print("Simulando ponto %d/%d: Cm=%.3f, Ch=%.3f, Bm=%.3f" % (i+1, N_PONTOS_TESTE, cm_val, ch_val, bm_val))
        resultado = skin_sim._calculate_pixel_color(parametros_pixel)
        
        r, g, b = resultado['rgb_float']
        cores_gt_rgb[i] =[r * 255.0, g * 255.0, b * 255.0]
        print("Resultado: R=%.2f, G=%.2f, B=%.2f" % (cores_gt_rgb[i, 0], cores_gt_rgb[i, 1], cores_gt_rgb[i, 2]))
        
    print(f"Simulações concluídas em {time.time() - start_time:.2f} segundos.")

    # 7. Calcular o Erro Perceptual (Delta E CIEDE2000)
    lut_formatado = (cores_lut_rgb / 255.0).reshape(-1, 1, 3)
    gt_formatado = (cores_gt_rgb / 255.0).reshape(-1, 1, 3)

    lab_lut = rgb2lab(lut_formatado)
    lab_gt = rgb2lab(gt_formatado)

    erros_de = deltaE_ciede2000(lab_gt, lab_lut).flatten()

    media_de = np.mean(erros_de)
    max_de = np.max(erros_de)

    print("\n--- Resultados Finais da Validação ---")
    print(f"Erro Médio (ΔE00): {media_de:.4f}")
    print(f"Erro Máximo (ΔE00): {max_de:.4f}")
    if media_de < 1.0:
        print("Veredito: SUCESSO! A discretização da LUT é densa o suficiente. A diferença é invisível ao olho humano.")
    else:
        print("Veredito: AVISO. A LUT apresenta perdas perceptíveis. Pode ser necessário aumentar a resolução.")

    # 8. Plotar o Histograma do Erro
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(erros_de, bins=20, color='darkcyan', edgecolor='black', alpha=0.8)

    plt.axvline(x=1.0, color='red', linestyle='dashed', linewidth=2, label='Limiar de Percepção Visual (ΔE = 1.0)')
    plt.axvline(x=media_de, color='yellow', linestyle='solid', linewidth=2, label=f'Média do Erro (ΔE = {media_de:.2f})')

    plt.title('Erro de Discretização da LUT (Interpolação vs. Simulação Direta)', fontsize=14)
    plt.xlabel('Erro Perceptual CIEDE2000 (ΔE00)', fontsize=12)
    plt.ylabel('Quantidade de Pontos Testados', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Garante que o código só rode se o script for chamado diretamente
if __name__ == "__main__":
    run()
# --- END OF FILE analysis/validar_discretizacao_lut.py ---