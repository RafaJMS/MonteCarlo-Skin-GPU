# --- START OF FILE main_generator.py ---
# --- START OF FILE main_generator.py ---
import time
import math
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- ADICIONE ESTAS TRÊS LINHAS AQUI ---
# Descobre a pasta exata onde o main_generator.py está salvo (a raiz do projeto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Cria o caminho absoluto para a pasta de dados biológicos
DATA_DIR = os.path.join(BASE_DIR, "data", "BiologicalParameters")

from core.data_loader import load_spectral_data, load_cie_cmf
from core.color_science import spectrum_to_xyz, xyz_to_srgb
from core.optics import get_tissue_optical_properties
from core.monte_carlo_cpu import run_monte_carlo_cpu

CUDA_AVAILABLE = False
try:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from core.monte_carlo_gpu import monte_carlo_reflectance_kernel_batch
    if cuda.is_available():
        cuda.select_device(0)
        CUDA_AVAILABLE = True
except Exception:
    pass

# --- Parâmetros Físicos e Configurações ---
N_PHOTONS = 100000
DEFAULT_SO2 = 0.75
DEFAULT_TEPI_MM = 0.25
WAVELENGTH_START = 380
WAVELENGTH_END = 780
WAVELENGTH_STEP = 10
G_ANISOTROPY = 0.9

MC_PI = np.pi
MC_ALIVE = 1
MC_DEAD = 0
MC_THRESHOLD = 0.01
MC_CHANCE = 0.1
MC_N_MEDIUM = 1.4
MC_N_AIR = 1.0 
FMEL_REF = 0.01

class SkinSimulation:
    def __init__(self, mode="GPU"):
        print(f"\n--- Inicializando Simulador de Pele ({mode}) ---")
        
        self.mode = mode
        if self.mode == "GPU" and not CUDA_AVAILABLE:
            print("AVISO: CUDA não detectado. Mudando automaticamente para modo CPU.")
            self.mode = "CPU"
            
        self.wavelengths_nm = np.arange(WAVELENGTH_START, WAVELENGTH_END + WAVELENGTH_STEP, WAVELENGTH_STEP)
        self.num_wls = len(self.wavelengths_nm)
        
        # Carrega funções CMF
        self.interp_cmf_x, self.interp_cmf_y, self.interp_cmf_z = load_cie_cmf(os.path.join(DATA_DIR, "CIE_xyz_1931_2deg.csv"))
        self.cmf_x = self.interp_cmf_x(self.wavelengths_nm)
        self.cmf_y = self.interp_cmf_y(self.wavelengths_nm)
        self.cmf_z = self.interp_cmf_z(self.wavelengths_nm)
        
        # Carrega e interpola Iluminante e Cromóforos
        self.interp_illum_d65 = load_spectral_data(os.path.join(DATA_DIR, "CIE_std_illum_D65.csv"))
        self.illum_d65 = self.interp_illum_d65(self.wavelengths_nm)
        
        integrand_Y_white = self.illum_d65 * self.cmf_y
        self.Y_white = np.trapezoid(integrand_Y_white, self.wavelengths_nm)
        if self.Y_white <= 1e-9: self.Y_white = 1.0

        self.hb_molar_conc_in_blood = 0.00237
        self.hba_spec = load_spectral_data(os.path.join(DATA_DIR, "hba_cm_inv_M_inv.csv"))(self.wavelengths_nm)
        self.hbo2a_spec = load_spectral_data(os.path.join(DATA_DIR, "hbo2a_cm_inv_M_inv.csv"))(self.wavelengths_nm)
        self.eu_ua_spec = load_spectral_data(os.path.join(DATA_DIR, "melanina_cm_inv.csv"))(self.wavelengths_nm)
        self.pheo_ua_spec = load_spectral_data(os.path.join(DATA_DIR, "pheomelanin_ua_data.csv"))(self.wavelengths_nm)
        
        if self.mode == "GPU":
            self.threads_per_wl = 1024 
            self.total_threads = self.num_wls * self.threads_per_wl
            self.rng_states = create_xoroshiro128p_states(self.total_threads, seed=42)
            print(f"CUDA Configurado: {self.num_wls} WLs, {self.threads_per_wl} threads/WL.")
            
        elif self.mode == "CPU":
            # Aquecimento do JIT para CPU
            print("Aquecendo compilador JIT (CPU)...")
            _ = run_monte_carlo_cpu(0.1, 10.0, 0.1, 10.0, 0.006, 10, G_ANISOTROPY, MC_PI, MC_ALIVE, MC_DEAD, MC_THRESHOLD, MC_CHANCE, MC_N_MEDIUM, MC_N_AIR)

    def _run_pixel_gpu(self, mua_e, mus_e, mua_d, mus_d, tepi_cm, n_photons):
        """Dispara as threads na Placa de Vídeo (Calcula todo o espectro em paralelo)"""
        photons_per_thread = math.ceil(n_photons / self.threads_per_wl)
        actual_photons_per_wl = photons_per_thread * self.threads_per_wl
        
        out_reflectance_gpu = cuda.to_device(np.zeros(self.num_wls, dtype=np.float64))
        d_mua_epi = cuda.to_device(mua_e)
        d_mus_epi = cuda.to_device(mus_e)
        d_mua_derm = cuda.to_device(mua_d)
        d_mus_derm = cuda.to_device(mus_d)
        
        threads_per_block = 256
        blocks_per_grid = (self.total_threads + (threads_per_block - 1)) // threads_per_block
        
        monte_carlo_reflectance_kernel_batch[blocks_per_grid, threads_per_block](
            out_reflectance_gpu, self.rng_states,
            d_mua_epi, d_mus_epi, d_mua_derm, d_mus_derm, tepi_cm,
            photons_per_thread, self.threads_per_wl, self.num_wls, G_ANISOTROPY,
            MC_PI, MC_ALIVE, MC_DEAD, MC_THRESHOLD, MC_CHANCE, MC_N_MEDIUM, MC_N_AIR
        )
        cuda.synchronize()
        return out_reflectance_gpu.copy_to_host() / actual_photons_per_wl

    def _run_pixel_cpu(self, mua_e, mus_e, mua_d, mus_d, tepi_cm, n_photons):
        """Roda a simulação no Processador, iterando cada comprimento de onda."""
        current_reflectance_spectrum = np.zeros(self.num_wls, dtype=np.float64)
        for i in range(self.num_wls):
            current_reflectance_spectrum[i] = run_monte_carlo_cpu(
                mua_e[i], mus_e[i], mua_d[i], mus_d[i], tepi_cm,
                n_photons, G_ANISOTROPY, MC_PI, MC_ALIVE, MC_DEAD, 
                MC_THRESHOLD, MC_CHANCE, MC_N_MEDIUM, MC_N_AIR
            )
        return current_reflectance_spectrum

    def _calculate_pixel_color(self, task_params):
        fmel, fblood, Bm, So2_sim, tepi_cm_sim, n_photons_sim, idx_Cm, idx_Ch, idx_Bm = task_params
        
        # Pega as propriedades ópticas da Pele usando nossa nova função isolada no módulo optics.py
        mua_e, mus_e, mua_d, mus_d = get_tissue_optical_properties(
            self.wavelengths_nm, fmel, fblood, Bm, So2_sim,
            self.eu_ua_spec, self.pheo_ua_spec, self.hba_spec, self.hbo2a_spec,
            FMEL_REF, self.hb_molar_conc_in_blood, G_ANISOTROPY
        )
        
        # Roteará para CPU ou GPU
        if self.mode == "GPU":
            spectrum = self._run_pixel_gpu(mua_e, mus_e, mua_d, mus_d, tepi_cm_sim, n_photons_sim)
        else:
            spectrum = self._run_pixel_cpu(mua_e, mus_e, mua_d, mus_d, tepi_cm_sim, n_photons_sim)

        # Transforma o Espectro de luz em Cor RGB usando o módulo color_science.py
        X, Y, Z = spectrum_to_xyz(spectrum, self.wavelengths_nm, self.illum_d65, self.cmf_x, self.cmf_y, self.cmf_z, self.Y_white)
        r_int, g_int, b_int = xyz_to_srgb(X, Y, Z, return_float=False)
        r_fl, g_fl, b_fl = xyz_to_srgb(X, Y, Z, return_float=True)

        return {
            'idx_Cm': idx_Cm, 'idx_Ch': idx_Ch, 'idx_Bm': idx_Bm,
            'Cm_fmel': fmel, 'Ch_fblood': fblood, 'Bm': Bm,
            'R_int': r_int, 'G_int': g_int, 'B_int': b_int,
            'rgb_float': (r_fl, g_fl, b_fl)
        }

    def generate_lut(self, Cm_vals, Ch_vals, Bm_vals, output_type="3D"):
        """Função unificada que gera a LUT (2D ou 3D)"""
        tepi_cm = DEFAULT_TEPI_MM / 10.0
        
        # =======================================================
        # MODO 2D: Calcula e salva um por um (3 barras de progresso)
        # =======================================================
        if output_type == "2D":
            for b_idx, bm in enumerate(Bm_vals):
                tasks =[]
                for y_idx, ch in enumerate(Ch_vals):
                    for x_idx, cm in enumerate(Cm_vals):
                        tasks.append((cm, ch, bm, DEFAULT_SO2, tepi_cm, N_PHOTONS, x_idx, y_idx, b_idx))
                
                results =[]
                print(f"\n--- Gerando LUT 2D para Bm = {bm:.2f} ({len(tasks)} pixels) ---")
                
                if self.mode == "CPU":
                    num_workers = os.cpu_count() or 4
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        futures = {executor.submit(self._calculate_pixel_color, t): t for t in tasks}
                        for f in tqdm(as_completed(futures), total=len(tasks), desc=f"CPU Bm={bm:.2f}"):
                            try: results.append(f.result())
                            except Exception as e: print(f"Erro no pixel: {e}")
                else:
                    for t in tqdm(tasks, desc=f"GPU Bm={bm:.2f}"):
                        try: results.append(self._calculate_pixel_color(t))
                        except Exception as e: print(f"Erro no pixel: {e}")
                
                # Ordena e salva PNG e CSV para este Bm
                results.sort(key=lambda r: (r['idx_Ch'], r['idx_Cm']))
                suffix = f"Bm_{bm:.2f}".replace('.', '_')
                
                # Salva Imagem
                img = Image.new('RGB', (len(Cm_vals), len(Ch_vals)), color='black')
                pixels = img.load()
                for r in results:
                    pixels[r['idx_Cm'], r['idx_Ch']] = (r['R_int'], r['G_int'], r['B_int'])
                img.save(f"lut_2D_{suffix}_{self.mode}.png")
                
                # Salva CSV
                csv_path = f"lut_data_2D_{suffix}_{self.mode}.csv"
                df = pd.DataFrame([{k: v for k, v in r.items() if k != 'rgb_float'} for r in results])
                df.to_csv(csv_path, index=False)
                print(f"Salvo: Imagem e CSV para Bm={bm:.2f}")

        # =======================================================
        # MODO 3D: Calcula tudo junto para formar o .cube
        # =======================================================
        elif output_type == "3D":
            tasks =[]
            for b_idx, bm in enumerate(Bm_vals):
                for y_idx, ch in enumerate(Ch_vals):
                    for x_idx, cm in enumerate(Cm_vals):
                        tasks.append((cm, ch, bm, DEFAULT_SO2, tepi_cm, N_PHOTONS, x_idx, y_idx, b_idx))
            
            results =[]
            print(f"\n--- Gerando LUT 3D completa ({len(tasks)} pixels) ---")
            
            if self.mode == "CPU":
                num_workers = os.cpu_count() or 4
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(self._calculate_pixel_color, t): t for t in tasks}
                    for f in tqdm(as_completed(futures), total=len(tasks), desc="CPU 3D"):
                        try: results.append(f.result())
                        except Exception as e: print(f"Erro no pixel: {e}")
            else:
                for t in tqdm(tasks, desc="GPU 3D"):
                    try: results.append(self._calculate_pixel_color(t))
                    except Exception as e: print(f"Erro no pixel: {e}")
            
            # Ordena e salva CUBE e CSV Geral
            results.sort(key=lambda r: (r['idx_Bm'], r['idx_Ch'], r['idx_Cm']))
            
            cube_path = f"lut_3D_{self.mode}.cube"
            with open(cube_path, 'w') as f:
                f.write(f'TITLE "Skin LUT ({self.mode})"\nLUT_3D_SIZE {len(Cm_vals)} {len(Ch_vals)} {len(Bm_vals)}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
                for r in results: f.write(f"{r['rgb_float'][0]:.6f} {r['rgb_float'][1]:.6f} {r['rgb_float'][2]:.6f}\n")
            print(f"Salvo: {cube_path}")
            
            csv_path = f"lut_data_3D_{self.mode}.csv"
            df = pd.DataFrame([{k: v for k, v in r.items() if k != 'rgb_float'} for r in results])
            df.to_csv(csv_path, index=False)
            print(f"Salvo: {csv_path}")

if __name__ == "__main__":
    
    fmel_range = np.geomspace(0.002, 0.5, 16)
    fblood_range = np.geomspace(0.003, 0.32, 12)
    Bm_range = np.array([0.01, 0.5, 0.99])
    
    # 1. Escolha o motor: "GPU" (Recomendado/Rápido) ou "CPU" (Plano B universal)
    ENGINE = "GPU"
    
    # 2. Escolha o output: "2D" ou "3D"
    OUTPUT_FORMAT = "3D"

    start_time = time.time()
    
    num_pixels_total = len(fmel_range) * len(fblood_range) * len(Bm_range)
    print("="*50); print("CONFIGURAÇÃO DA SIMULAÇÃO:")
    print(f"  N_PHOTONS: {N_PHOTONS}"); print(f"  WAVELENGTH_STEP: {WAVELENGTH_STEP}nm")
    print(f"  Cm_range (análogo fmel) terá {len(fmel_range)} valores.")
    print(f"  Ch_range (análogo fblood) terá {len(fblood_range)} valores.")
    print(f"  Bm_range terá {len(Bm_range)} valores: {Bm_range}");
    print(f"  Total de combinações/pixels = {num_pixels_total}");
    print(f"  Dimensões da LUT: {len(fmel_range) * len(Bm_range)} L x {len(fblood_range)} A")
    print("="*50)

    simulador = SkinSimulation(mode=ENGINE)
    simulador.generate_lut(fmel_range, fblood_range, Bm_range, output_type=OUTPUT_FORMAT)

    print(f"\nFinalizado em {(time.time() - start_time)/60.0:.2f} minutos!")