import time
import math
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

N_PHOTONS = 100000
DEFAULT_SO2 = 0.75
DEFAULT_TEPI_MM = 0.25
WAVELENGTH_START = 380
WAVELENGTH_END = 780
WAVELENGTH_STEP = 5
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
            self.threads_per_wl = 2048
            self.total_threads = self.num_wls * self.threads_per_wl
            self.rng_states = create_xoroshiro128p_states(self.total_threads, seed=42)
            
            # Pré-aloca buffers uma única vez
            self._gpu_out      = cuda.device_array(self.num_wls, dtype=np.float64)
            self._gpu_mua_epi  = cuda.device_array(self.num_wls, dtype=np.float64)
            self._gpu_mus_epi  = cuda.device_array(self.num_wls, dtype=np.float64)
            self._gpu_mua_derm = cuda.device_array(self.num_wls, dtype=np.float64)
            self._gpu_mus_derm = cuda.device_array(self.num_wls, dtype=np.float64)
            
            # Calcula configuração de blocos uma única vez
            self._threads_per_block = 512
            self._blocks_per_grid = (
                self.total_threads + self._threads_per_block - 1
            ) // self._threads_per_block
            
            import time
            import warnings
            from numba.core.errors import NumbaPerformanceWarning

            configs = [
                (512,  256),
                (1024, 256),
                (1024, 512),
                (2048, 256),
                (2048, 512),
            ]
            dummy = np.ones(self.num_wls, dtype=np.float64) * 0.1

            melhor_tempo  = float('inf')
            melhor_config = configs[0]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
                
                for tpw, tpb in configs:
                    total  = self.num_wls * tpw
                    blocos = (total + tpb - 1) // tpb

                    # Aquece
                    self._run_pixel_gpu(dummy, dummy, dummy, dummy, 0.006, 1000)

                    # Mede
                    t0 = time.time()
                    for _ in range(5):
                        self._run_pixel_gpu(dummy, dummy, dummy, dummy, 0.025, 100000)
                    tempo = (time.time() - t0) / 5

                    print(f"threads_per_wl={tpw:4d}, threads_per_block={tpb}: {tempo:.4f}s/pixel  ({blocos} blocos)")

                    if tempo < melhor_tempo:
                        melhor_tempo  = tempo
                        melhor_config = (tpw, tpb)

            tpw_best, tpb_best = melhor_config
            self.threads_per_wl     = tpw_best
            self.total_threads      = self.num_wls * tpw_best
            self._threads_per_block = tpb_best
            self._blocks_per_grid   = (self.total_threads + tpb_best - 1) // tpb_best
            self.rng_states         = create_xoroshiro128p_states(self.total_threads, seed=42)

            print(f"\n-> Configuração selecionada: threads_per_wl={tpw_best}, threads_per_block={tpb_best} ({self._blocks_per_grid} blocos)")
            
        elif self.mode == "CPU":
            print("Aquecendo compilador JIT (CPU)...")
            _ = run_monte_carlo_cpu(0.1, 10.0, 0.1, 10.0, 0.006, 10, G_ANISOTROPY, MC_PI, MC_ALIVE, MC_DEAD, MC_THRESHOLD, MC_CHANCE, MC_N_MEDIUM, MC_N_AIR)

    def _run_pixel_gpu(self, mua_e, mus_e, mua_d, mus_d, tepi_cm, n_photons):
        photons_per_thread = math.ceil(n_photons / self.threads_per_wl)
        actual_photons_per_wl = photons_per_thread * self.threads_per_wl

        # Zera e copia nos buffers pré-alocados — sem alocação nova
        self._gpu_out[:] = 0.0
        self._gpu_mua_epi.copy_to_device(mua_e)
        self._gpu_mus_epi.copy_to_device(mus_e)
        self._gpu_mua_derm.copy_to_device(mua_d)
        self._gpu_mus_derm.copy_to_device(mus_d)

        monte_carlo_reflectance_kernel_batch[
            self._blocks_per_grid,
            self._threads_per_block
        ](
            self._gpu_out, self.rng_states,
            self._gpu_mua_epi, self._gpu_mus_epi,
            self._gpu_mua_derm, self._gpu_mus_derm,
            tepi_cm, photons_per_thread, self.threads_per_wl,
            self.num_wls, G_ANISOTROPY,
            MC_PI, MC_ALIVE, MC_DEAD, MC_THRESHOLD, MC_CHANCE,
            MC_N_MEDIUM, MC_N_AIR
        )
        cuda.synchronize()
        return self._gpu_out.copy_to_host() / actual_photons_per_wl

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
        
        mua_e, mus_e, mua_d, mus_d = get_tissue_optical_properties(
            self.wavelengths_nm, fmel, fblood, Bm, So2_sim,
            self.eu_ua_spec, self.pheo_ua_spec, self.hba_spec, self.hbo2a_spec,
            FMEL_REF, self.hb_molar_conc_in_blood, G_ANISOTROPY
        )
        
        if self.mode == "GPU":
            spectrum = self._run_pixel_gpu(mua_e, mus_e, mua_d, mus_d, tepi_cm_sim, n_photons_sim)
        else:
            spectrum = self._run_pixel_cpu(mua_e, mus_e, mua_d, mus_d, tepi_cm_sim, n_photons_sim)

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
                
                results.sort(key=lambda r: (r['idx_Ch'], r['idx_Cm']))
                suffix = f"Bm_{bm:.2f}".replace('.', '_')
                
                img = Image.new('RGB', (len(Cm_vals), len(Ch_vals)), color='black')
                pixels = img.load()
                for r in results:
                    pixels[r['idx_Cm'], r['idx_Ch']] = (r['R_int'], r['G_int'], r['B_int'])
                img.save(f"lut_2D_{suffix}_{self.mode}.png")
                
                csv_path = f"lut_data_2D_{suffix}_{self.mode}.csv"
                df = pd.DataFrame([{k: v for k, v in r.items() if k != 'rgb_float'} for r in results])
                df.to_csv(csv_path, index=False)
                print(f"Salvo: Imagem e CSV para Bm={bm:.2f}")

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
    OUTPUT_FORMAT = "2D"

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
