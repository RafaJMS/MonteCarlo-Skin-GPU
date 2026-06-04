import numpy as np
from numba import njit

@njit()
def calculate_fresnel_cpu(n1, n2, cos_theta1):
    if cos_theta1 > 1.0: cos_theta1 = 1.0
    elif cos_theta1 < -1.0: cos_theta1 = -1.0

    if n1 == n2: return 0.0

    sin_theta1_sq = 1.0 - cos_theta1 * cos_theta1
    if sin_theta1_sq < 0.0: sin_theta1_sq = 0.0

    ratio = n1 / n2
    sin_theta2_sq = (ratio * ratio) * sin_theta1_sq

    if sin_theta2_sq >= 1.0: return 1.0

    cos_theta2 = np.sqrt(1.0 - sin_theta2_sq)

    rs_den = (n1 * cos_theta1 + n2 * cos_theta2)
    rp_den = (n2 * cos_theta1 + n1 * cos_theta2)
    
    rs = ((n1 * cos_theta1 - n2 * cos_theta2) / rs_den) ** 2 if rs_den != 0.0 else 1.0
    rp = ((n2 * cos_theta1 - n1 * cos_theta2) / rp_den) ** 2 if rp_den != 0.0 else 1.0

    return 0.5 * (rs + rp)

@njit()
def run_monte_carlo_cpu(
    mua_epi, mus_epi, mua_derm, mus_derm, tepi_cm,
    n_photons_mc, g_anisotropy,
    mc_pi, mc_alive, mc_dead, mc_threshold, mc_chance,
    mc_n_medium, mc_n_air
):
    eps = 1e-12
    mut_epi = mua_epi + mus_epi
    mut_derm = mua_derm + mus_derm
    albedo_epi = mus_epi / mut_epi if mut_epi > eps else 0.0
    albedo_derm = mus_derm / mut_derm if mut_derm > eps else 0.0

    R_spec = 0.0
    if mc_n_medium != mc_n_air:
        R_spec = ((mc_n_air - mc_n_medium) / (mc_n_air + mc_n_medium)) ** 2

    total_reflectance_weight = 0.0

    for p in range(n_photons_mc):
        W = 1.0 - R_spec
        x = 0.0; y = 0.0; z = 0.0
        ux = 0.0; uy = 0.0; uz = 1.0

        current_layer_is_epi = True
        alive = True

        while alive:
            # 1. Definir o passo óptico
            xi = np.random.rand()
            while xi <= 0.0: xi = np.random.rand()
            s_opt = -np.log(xi)

            # 2. Loop de Fronteiras (Movimenta o fóton até esgotar o passo)
            while s_opt > 0.0 and alive:
                mut = mut_epi if current_layer_is_epi else mut_derm
                tau_b = np.inf
                boundary_type = 0

                if current_layer_is_epi:
                    if uz < -eps:
                        dist_cm = z / (-uz)
                        if dist_cm > eps:
                            tau_b = mut * dist_cm
                            boundary_type = 1
                    if uz > eps:
                        dist_cm = (tepi_cm - z) / uz
                        if dist_cm > eps:
                            tb = mut * dist_cm
                            if tb < tau_b:
                                tau_b = tb
                                boundary_type = 2
                else:
                    if uz < -eps:
                        dist_cm = (z - tepi_cm) / (-uz)
                        if dist_cm > eps:
                            tau_b = mut * dist_cm
                            boundary_type = 2

                if s_opt > tau_b:
                    step_cm = tau_b / (mut if mut > eps else 1.0)
                    x += step_cm * ux; y += step_cm * uy; z += step_cm * uz
                    s_opt -= tau_b

                    if boundary_type == 1:
                        R_int = calculate_fresnel_cpu(mc_n_medium, mc_n_air, abs(uz))
                        if np.random.rand() > R_int:
                            total_reflectance_weight += W
                            alive = False
                            break
                        else:
                            uz = -uz; z = 1e-12
                    elif boundary_type == 2:
                        current_layer_is_epi = not current_layer_is_epi
                        if current_layer_is_epi: z = tepi_cm - 1e-12 if uz < 0.0 else tepi_cm + 1e-12
                        else: z = tepi_cm + 1e-12 if uz > 0.0 else tepi_cm - 1e-12
                else:
                    step_cm = s_opt / (mut if mut > eps else 1.0)
                    x += step_cm * ux; y += step_cm * uy; z += step_cm * uz
                    s_opt = 0.0

            if not alive:
                break

            # 3. Absorção, Espalhamento (H-G) e Roleta - Isolados corretamente aqui
            albedo = albedo_epi if current_layer_is_epi else albedo_derm
            W *= albedo

            rnd = np.random.rand()
            if abs(g_anisotropy) < 1e-12: cost = 2.0 * rnd - 1.0
            else:
                tmp = (1.0 - g_anisotropy * g_anisotropy) / (1.0 - g_anisotropy + 2.0 * g_anisotropy * rnd)
                cost = (1.0 + g_anisotropy * g_anisotropy - tmp * tmp) / (2.0 * g_anisotropy)
                cost = min(max(cost, -1.0), 1.0)

            sint = np.sqrt(max(0.0, 1.0 - cost * cost))
            phi = 2.0 * mc_pi * np.random.rand()
            cphi = np.cos(phi); sphi = np.sin(phi)

            if abs(uz) > 0.99999:
                uxx = sint * cphi; uyy = sint * sphi; uzz = cost * (1.0 if uz > 0.0 else -1.0)
            else:
                denom = np.sqrt(max(0.0, 1.0 - uz * uz))
                uxx = sint * (ux * uz * cphi - uy * sphi) / denom + ux * cost
                uyy = sint * (uy * uz * cphi + ux * sphi) / denom + uy * cost
                uzz = -sint * cphi * denom + uz * cost

            norm = np.sqrt(uxx * uxx + uyy * uyy + uzz * uzz)
            if norm > 0.0:
                ux = uxx / norm; uy = uyy / norm; uz = uzz / norm
            else:
                ux = 0.0; uy = 0.0; uz = 1.0

            # Roleta Russa
            if W < mc_threshold:
                if np.random.rand() <= mc_chance:
                    W /= mc_chance
                else:
                    alive = False

    return total_reflectance_weight / n_photons_mc if n_photons_mc > 0 else 0.0
