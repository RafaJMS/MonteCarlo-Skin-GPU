import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
 
# Garante que o Python ache a pasta 'core' na raiz do projeto
raiz_projeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, raiz_projeto)
 
# Importa a nossa física óptica pura!
from core.optics import get_tissue_optical_properties
from core.monte_carlo_cpu import run_monte_carlo_cpu
from main_generator import load_spectral_data, FMEL_REF, G_ANISOTROPY, N_PHOTONS
 
# Tenta importar a GPU
CUDA_AVAILABLE = False
try:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from core.monte_carlo_gpu import monte_carlo_reflectance_kernel_batch
    if cuda.is_available():
        CUDA_AVAILABLE = True
except Exception:
    pass
 
def run():
    print("--- Iniciando Benchmarking HPC (CPU vs GPU) ---")
 
    if not CUDA_AVAILABLE:
        print("Erro: CUDA não detectado. O teste de Speedup precisa de uma GPU NVIDIA.")
        return
 
    # Especifica o hardware usado (importante para o artigo IEEE)
    gpu_name = cuda.get_current_device().name.decode()
    print(f"GPU detectada: {gpu_name}")
    print(f"CPU: {os.popen('cat /proc/cpuinfo | grep \"model name\" | head -1').read().strip()}")
 
    # Parâmetros físicos fixos para o teste
    fmel, fblood, Bm, So2 = 0.05, 0.03, 0.5, 0.75
    tepi_cm = 0.025
 
    DATA_DIR = os.path.join(raiz_projeto, "data", "BiologicalParameters")
    hba_interp   = load_spectral_data(os.path.join(DATA_DIR, "hba_cm_inv_M_inv.csv"))
    hbo2a_interp = load_spectral_data(os.path.join(DATA_DIR, "hbo2a_cm_inv_M_inv.csv"))
    eu_interp    = load_spectral_data(os.path.join(DATA_DIR, "melanina_cm_inv.csv"))
    pheo_interp  = load_spectral_data(os.path.join(DATA_DIR, "pheomelanin_ua_data.csv"))
 
    # Resoluções espectrais a testar (de 40 nm rápido até 2 nm pesado)
    test_steps_nm = [40, 20, 10, 5, 2]
    N_RUNS = 5  # número de repetições para média robusta
 
    wavelength_counts = []
    cpu_times = []
    gpu_times = []
 
    # Aquecimento JIT (evita penalizar CPU/GPU na primeira execução)
    print("\nAquecendo os motores...")
    _ = run_monte_carlo_cpu(
        0.1, 10.0, 0.1, 10.0, 0.006, 10,
        G_ANISOTROPY, np.pi, 1, 0, 0.01, 0.1, 1.33, 1.0
    )
 
    for step in test_steps_nm:
        wls_nm  = np.arange(380, 780 + step, step)
        num_wls = len(wls_nm)
        wavelength_counts.append(num_wls)
        print(f"\n[Testando para {num_wls} comprimentos de onda (step {step} nm)]")
 
        # Prepara a óptica
        mua_e, mus_e, mua_d, mus_d = get_tissue_optical_properties(
            wls_nm, fmel, fblood, Bm, So2,
            eu_interp(wls_nm), pheo_interp(wls_nm),
            hba_interp(wls_nm), hbo2a_interp(wls_nm),
            FMEL_REF, 0.00237, G_ANISOTROPY
        )
 
        # --- CPU: média de N_RUNS execuções, descarta a primeira (warm-up) ---
        run_times_cpu = []
        for r in range(N_RUNS):
            start = time.time()
            for i in range(num_wls):
                _ = run_monte_carlo_cpu(
                    mua_e[i], mus_e[i], mua_d[i], mus_d[i],
                    tepi_cm, N_PHOTONS, G_ANISOTROPY,
                    np.pi, 1, 0, 0.01, 0.1, 1.4, 1.0
                )
            run_times_cpu.append(time.time() - start)
        time_cpu = np.mean(run_times_cpu[1:])  # descarta warm-up
        cpu_times.append(time_cpu)
        print(f"  -> Tempo CPU (média {N_RUNS-1} runs): {time_cpu:.4f} s")
 
        # --- GPU: média de N_RUNS execuções, descarta a primeira (warm-up) ---
        threads_per_wl    = 1024
        total_threads     = num_wls * threads_per_wl
        photons_per_thread = math.ceil(N_PHOTONS / threads_per_wl)
        threads_per_block = 256
        blocks_per_grid   = (total_threads + (threads_per_block - 1)) // threads_per_block
 
        run_times_gpu = []
        for r in range(N_RUNS):
            rng_states   = create_xoroshiro128p_states(total_threads, seed=42)
            out_gpu      = cuda.to_device(np.zeros(num_wls, dtype=np.float64))
            d_mua_epi    = cuda.to_device(mua_e)
            d_mus_epi    = cuda.to_device(mus_e)
            d_mua_derm   = cuda.to_device(mua_d)
            d_mus_derm   = cuda.to_device(mus_d)
 
            start = time.time()
            monte_carlo_reflectance_kernel_batch[blocks_per_grid, threads_per_block](
                out_gpu, rng_states,
                d_mua_epi, d_mus_epi, d_mua_derm, d_mus_derm,
                tepi_cm, photons_per_thread, threads_per_wl, num_wls,
                G_ANISOTROPY, np.pi, 1, 0, 0.01, 0.1, 1.4, 1.0
            )
            cuda.synchronize()
            run_times_gpu.append(time.time() - start)
        time_gpu = np.mean(run_times_gpu[1:])  # descarta warm-up
        gpu_times.append(time_gpu)
        print(f"  -> Tempo GPU (média {N_RUNS-1} runs): {time_gpu:.4f} s")
 
    # --- Speedup consolidado ---
    speedup = np.array(cpu_times) / np.array(gpu_times)
 
    idx_5nm      = test_steps_nm.index(5)
    speedup_5nm  = speedup[idx_5nm]
    print(f"\n=== SPEEDUP CONSOLIDADO (5 nm, resolução normativa): {speedup_5nm:.1f}x ===")
 
    # --- Gráfico 1: Tempos absolutos ---
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength_counts, cpu_times, '-o',
             label='CPU (Numba JIT)', color='blue', linewidth=2, markersize=8)
    plt.plot(wavelength_counts, gpu_times, '-x',
             label='GPU (CUDA/Numba)', color='red', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pontos de comprimento de onda', fontsize=12)
    plt.ylabel('Tempo de execução (s)', fontsize=12)
    plt.title('Benchmarking de performance: CPU vs. GPU', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
 
    # --- Gráfico 2: Speedup ---
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength_counts, speedup, '-s',
             color='green', linewidth=2, markersize=8)
    plt.axvline(x=wavelength_counts[idx_5nm], color='gray',
                linestyle='--', linewidth=1.5, label='5 nm (resolução normativa)')
    plt.annotate(f'{speedup_5nm:.1f}×',
                 xy=(wavelength_counts[idx_5nm], speedup_5nm),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=11, color='green')
    plt.xscale('log')
    plt.xlabel('Pontos de comprimento de onda', fontsize=12)
    plt.ylabel('Speedup (tempo CPU / tempo GPU)', fontsize=12)
    plt.title('Ganho de velocidade da GPU vs. complexidade do espectro', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
 
if __name__ == "__main__":
    run()
 
