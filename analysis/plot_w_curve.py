# --- START OF FILE analysis/plot_w_curve.py ---
import matplotlib.pyplot as plt

# Importa o simulador que você já construiu na raiz do projeto!
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_generator import SkinSimulation, DEFAULT_SO2, DEFAULT_TEPI_MM

def run():
    print("--- Gerando Curva de Validação Espectral (Curva em 'W') ---")
    
    simulador = SkinSimulation(mode="GPU")
    
    fmel, fblood, Bm = 0.00, 0.02, 0.50
    tepi_cm = DEFAULT_TEPI_MM / 10.0
    
    from core.optics import get_tissue_optical_properties
    from main_generator import FMEL_REF, G_ANISOTROPY, N_PHOTONS
    
    mua_e, mus_e, mua_d, mus_d = get_tissue_optical_properties(
        simulador.wavelengths_nm, fmel, fblood, Bm, DEFAULT_SO2,
        simulador.eu_ua_spec, simulador.pheo_ua_spec, simulador.hba_spec, simulador.hbo2a_spec,
        FMEL_REF, simulador.hb_molar_conc_in_blood, G_ANISOTROPY
    )
    
    if simulador.mode == "GPU":
        spectrum = simulador._run_pixel_gpu(mua_e, mus_e, mua_d, mus_d, tepi_cm, N_PHOTONS)
    else:
        spectrum = simulador._run_pixel_cpu(mua_e, mus_e, mua_d, mus_d, tepi_cm, N_PHOTONS)

    plt.figure(figsize=(10, 6))
    plt.plot(simulador.wavelengths_nm, spectrum, color='purple', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
    plt.ylabel('Refletância Bruta', fontsize=12)
    plt.title(f'Assinatura Espectral da Pele (Curva em "W")\nCm={fmel:.2f}, Ch={fblood:.2f}, Bm={Bm:.2f}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvspan(530, 550, color='red', alpha=0.15, label='Pico HbO2 (~542nm)')
    plt.axvspan(565, 585, color='red', alpha=0.15, label='Pico HbO2 (~577nm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()