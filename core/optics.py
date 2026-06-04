import numpy as np
def get_tissue_optical_properties(
    wavelengths_nm, fmel, fblood, Bm, So2,
    eu_ua_spec, pheo_ua_spec, hba_spec, hbo2a_spec,
    fmel_ref=0.01, hb_molar_conc_in_blood=0.00237, g_anisotropy=0.9
):
    mua_baseline = np.full_like(wavelengths_nm, 0.05, dtype=np.float64)

    # --- Epiderme ---
    scaling_factor = fmel / fmel_ref if fmel_ref > 1e-9 else 0.0
    if fmel <= 1e-9: 
        scaling_factor = 0.0

    mua_mel_combined = scaling_factor * (Bm * eu_ua_spec + (1.0 - Bm) * pheo_ua_spec)

    mua_epi_arr = mua_mel_combined + mua_baseline 

    # --- Derme ---
    hb_molar_conc_tissue = fblood * hb_molar_conc_in_blood
    mua_blood = hb_molar_conc_tissue * (So2 * hbo2a_spec + (1.0 - So2) * hba_spec)

    mua_derm_arr = mua_blood + mua_baseline 

    # --- Espalhamento (mus) ---
    a_epi, b_epi = 60.0, 1.2
    a_derm, b_derm = 40.0, 1.0

    mus_prime_epi = a_epi * (wavelengths_nm / 500.0)**(-b_epi)
    mus_prime_derm = a_derm * (wavelengths_nm / 500.0)**(-b_derm)

    den_g = 1.0 - g_anisotropy
    if abs(den_g) > 1e-9:
        mus_epi_arr = mus_prime_epi / den_g
        mus_derm_arr = mus_prime_derm / den_g
    else:
        mus_epi_arr = mus_prime_epi * 1e9
        mus_derm_arr = mus_prime_derm * 1e9

    return mua_epi_arr.astype(np.float64), mus_epi_arr.astype(np.float64), mua_derm_arr.astype(np.float64), mus_derm_arr.astype(np.float64)