import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

def load_spectral_data(path, wavelength_col=0, value_col=1, skiprows=0, delimiter=',', is_custom_format=False):
    print(f"  Carregando dados {'(custom)' if is_custom_format else ''} de: {os.path.basename(path)}")
    load_delimiter = '\\s+' if is_custom_format else delimiter
    try:
        df = pd.read_csv(path, skiprows=skiprows, delimiter=load_delimiter, header=None, skipinitialspace=True, engine='python', on_bad_lines='warn')
        if df.shape[1] <= max(wavelength_col, value_col):
             raise ValueError(f"Arquivo tem menos colunas ({df.shape[1]}) do que o esperado (precisa de pelo menos {max(wavelength_col, value_col)+1}). Verifique delimitador, skiprows, header.")
        
        wavelengths_raw = pd.to_numeric(df[wavelength_col], errors='coerce')
        values_raw = pd.to_numeric(df[value_col], errors='coerce')
        
        valid_indices = ~np.isnan(wavelengths_raw) & ~np.isnan(values_raw)
        if not np.all(valid_indices):
            removed_count = np.sum(~valid_indices)
            print(f"    Atenção: Removendo {removed_count} linhas com valores não numéricos/NaN em {os.path.basename(path)}")
            wavelengths = wavelengths_raw[valid_indices].to_numpy(dtype=float)
            values = values_raw[valid_indices].to_numpy(dtype=float)
        else:
            wavelengths = wavelengths_raw.to_numpy(dtype=float)
            values = values_raw.to_numpy(dtype=float)
            
        if len(wavelengths) < 2:
            raise ValueError(f"Não há dados válidos suficientes ({len(wavelengths)}) após limpeza em {path}")
            
        sort_indices = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_indices]; values = values[sort_indices]
        
        unique_wavelengths, unique_indices = np.unique(wavelengths, return_index=True)
        if len(unique_wavelengths) < len(wavelengths):
             print(f"    Atenção: Removendo {len(wavelengths) - len(unique_wavelengths)} comprimentos de onda duplicados em {os.path.basename(path)}")
             wavelengths = wavelengths[unique_indices]; values = values[unique_indices]
             
        if len(wavelengths) < 2:
             raise ValueError(f"Não há dados válidos suficientes ({len(wavelengths)}) após remover duplicatas em {path}")
             
        interp_func = interp1d(wavelengths, values, kind='linear', bounds_error=False, fill_value=0.0)
        print(f"    Dados de {os.path.basename(path)} carregados. Faixa usada: {wavelengths.min()}-{wavelengths.max()}nm. Valores médios: {np.mean(values):.2e}")
        return interp_func
    except FileNotFoundError:
        print(f"Erro CRÍTICO: Arquivo não encontrado em '{path}'")
        raise
    except Exception as e:
        print(f"Erro CRÍTICO ao carregar ou processar {path}: {e}")
        print("         Verifique o caminho, formato do arquivo, delimitador, skiprows e índices das colunas.")
        raise

def load_cie_cmf(path):
    print(f"  Carregando CMF CIE de: {path}")
    try:
        df = pd.read_csv(path, header=None, skiprows=0, delimiter=',')
        if df.shape[1] < 4: raise ValueError("Arquivo CMF não tem 4 colunas.")
        
        wavelengths = pd.to_numeric(df[0], errors='coerce').to_numpy(dtype=float)
        x_bar = pd.to_numeric(df[1], errors='coerce').to_numpy(dtype=float)
        y_bar = pd.to_numeric(df[2], errors='coerce').to_numpy(dtype=float)
        z_bar = pd.to_numeric(df[3], errors='coerce').to_numpy(dtype=float)
        
        valid_indices = ~np.isnan(wavelengths); wavelengths = wavelengths[valid_indices]
        x_bar = x_bar[valid_indices]; y_bar = y_bar[valid_indices]; z_bar = z_bar[valid_indices]
        
        sort_indices = np.argsort(wavelengths); wavelengths = wavelengths[sort_indices]
        x_bar = x_bar[sort_indices]; y_bar = y_bar[sort_indices]; z_bar = z_bar[sort_indices]
        
        interp_x = interp1d(wavelengths, x_bar, kind='linear', bounds_error=False, fill_value=0.0)
        interp_y = interp1d(wavelengths, y_bar, kind='linear', bounds_error=False, fill_value=0.0)
        interp_z = interp1d(wavelengths, z_bar, kind='linear', bounds_error=False, fill_value=0.0)
        
        print(f"    CMFs CIE carregados. Faixa usada: {wavelengths.min()}-{wavelengths.max()}nm.")
        return interp_x, interp_y, interp_z
    except FileNotFoundError: 
        print(f"Erro CRÍTICO: Arquivo CMF não encontrado em '{path}'"); raise
    except Exception as e: 
        print(f"Erro CRÍTICO ao carregar dados CMF de {path}: {e}"); raise
