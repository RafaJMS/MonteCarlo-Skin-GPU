# --- START OF FILE core/color_science.py ---
import numpy as np

def spectrum_to_xyz(reflectance_spectrum, wavelengths_nm, illum_d65, cmf_x, cmf_y, cmf_z, Y_white):
    """
    Converte um espectro de refletância bruto para o espaço de cor CIE XYZ,
    utilizando a iluminação D65 e as funções de matching de cor (CMF).
    """
    radiance = reflectance_spectrum * illum_d65
    
    X = np.trapezoid(radiance * cmf_x, wavelengths_nm)
    Y = np.trapezoid(radiance * cmf_y, wavelengths_nm)
    Z = np.trapezoid(radiance * cmf_z, wavelengths_nm)
    
    norm_factor = 100.0 / Y_white if Y_white > 1e-9 else 1.0
    return X * norm_factor, Y * norm_factor, Z * norm_factor

def xyz_to_srgb(X, Y, Z, return_float=False):
    """
    Converte valores do espaço de cor CIE XYZ para o espaço padrão sRGB.
    Pode retornar inteiros (0-255) para imagens, ou floats (0.0-1.0) para matrizes LUT (.cube).
    """
    x, y, z_ = X / 100.0, Y / 100.0, Z / 100.0
    
    # Matriz de transformação linear XYZ para RGB
    mat_xyz_to_rgb = np.array([[ 3.2404542, -1.5371385, -0.4985314],[-0.9692660,  1.8760108,  0.0415560],[ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    r_lin, g_lin, b_lin = np.dot(mat_xyz_to_rgb, [x, y, z_])
    
    # Clamp dos valores para manter entre 0 e 1 antes da correção gama
    r_lin = max(0.0, min(1.0, r_lin))
    g_lin = max(0.0, min(1.0, g_lin))
    b_lin = max(0.0, min(1.0, b_lin))
    
    # Função de correção gama sRGB (não linear)
    srgb_gamma_func = np.vectorize(lambda c: 12.92 * c if c <= 0.0031308 else 1.055 * (c**(1.0/2.4)) - 0.055)
    r, g, b = srgb_gamma_func([r_lin, g_lin, b_lin])
    
    if return_float:
        return (r, g, b)
    
    # Retorna como inteiro 0-255 para formar a cor do pixel em imagens PNG
    return (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))
# --- END OF FILE core/color_science.py ---