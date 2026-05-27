import os
os.environ['NUMBA_FORCE_CUDA_CC'] = '8.9'
os.environ['NUMBA_CUDA_DEFAULT_PTX_CC'] = '8.9'

import math
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64

@cuda.jit(device=True)
def calculate_fresnel_gpu(n1, n2, cos_theta1):
    if cos_theta1 > 1.0: cos_theta1 = 1.0
    elif cos_theta1 < -1.0: cos_theta1 = -1.0
    
    if n1 == n2: return 0.0
    
    sin_theta1_sq = 1.0 - cos_theta1*cos_theta1
    if sin_theta1_sq < 0.0: sin_theta1_sq = 0.0
    
    sin_theta2_sq = ((n1 / n2)**2) * sin_theta1_sq
    if sin_theta2_sq >= 1.0: return 1.0
    
    cos_theta2 = math.sqrt(1.0 - sin_theta2_sq)
    
    r_s_num = n1 * cos_theta1 - n2 * cos_theta2
    r_s_den = n1 * cos_theta1 + n2 * cos_theta2
    r_s = (r_s_num / r_s_den)**2 if r_s_den != 0.0 else 1.0
    
    r_p_num = n2 * cos_theta1 - n1 * cos_theta2
    r_p_den = n2 * cos_theta1 + n1 * cos_theta2
    r_p = (r_p_num / r_p_den)**2 if r_p_den != 0.0 else 1.0
    
    return (r_s + r_p) / 2.0

@cuda.jit
def monte_carlo_reflectance_kernel_batch(
    out_reflectance_array, rng_states,
    mua_epi_arr, mus_epi_arr, mua_derm_arr, mus_derm_arr, tepi_cm,
    photons_per_thread, threads_per_wl, num_wls, g_anisotropy,
    mc_pi, mc_alive, mc_dead, mc_threshold, mc_chance,
    mc_n_medium, mc_n_air):
    
    thread_id = cuda.grid(1)
    if thread_id >= rng_states.shape[0]: return
    
    wl_idx = thread_id // threads_per_wl
    if wl_idx >= num_wls: return
    
    mua_epi = mua_epi_arr[wl_idx]
    mus_epi = mus_epi_arr[wl_idx]
    mua_derm = mua_derm_arr[wl_idx]
    mus_derm = mus_derm_arr[wl_idx]

    albedo_epi = mus_epi / (mua_epi + mus_epi) if (mua_epi + mus_epi) > 1e-9 else 0.0
    albedo_derm = mus_derm / (mua_derm + mus_derm) if (mua_derm + mus_derm) > 1e-9 else 0.0
    mut_epi = mua_epi + mus_epi
    mut_derm = mua_derm + mus_derm

    local_reflectance = 0.0

    for _ in range(photons_per_thread):
        W = 1.0; x = 0.0; y = 0.0; z = 0.0; ux = 0.0; uy = 0.0; uz = 1.0
        photon_status = mc_alive
        current_layer_is_epi = True
        R_specular = ((mc_n_air - mc_n_medium)/(mc_n_air + mc_n_medium))**2
        W *= (1.0 - R_specular)
        
        max_steps = 2000
        for step_count in range(max_steps):
            mua = mua_epi if current_layer_is_epi else mua_derm
            mut = mut_epi if current_layer_is_epi else mut_derm
            albedo = albedo_epi if current_layer_is_epi else albedo_derm
            
            rnd_step = xoroshiro128p_uniform_float64(rng_states, thread_id)
            while rnd_step <= 0.0: 
                rnd_step = xoroshiro128p_uniform_float64(rng_states, thread_id)
                
            s_total_step = -math.log(rnd_step) / mut if mut > 1e-9 else 1e10
            s_remaining = s_total_step
            
            while s_remaining > 1e-9:
                s_to_boundary = 1e10; boundary_type = 0
                if uz < 0.0:
                    s_to_top_candidate = (0.0 - z) / uz if abs(uz) > 1e-9 else 1e10
                    if s_to_top_candidate < s_to_boundary and s_to_top_candidate > 1e-9:
                        s_to_boundary = s_to_top_candidate; boundary_type = 1
                if current_layer_is_epi and uz > 0.0:
                    s_to_internal_candidate = (tepi_cm - z) / uz if abs(uz) > 1e-9 else 1e10
                    if s_to_internal_candidate < s_to_boundary and s_to_internal_candidate > 1e-9:
                        s_to_boundary = s_to_internal_candidate; boundary_type = 2
                elif not current_layer_is_epi and uz < 0.0:
                    s_to_internal_candidate = (tepi_cm - z) / uz if abs(uz) > 1e-9 else 1e10
                    if s_to_internal_candidate < s_to_boundary and s_to_internal_candidate > 1e-9:
                        s_to_boundary = s_to_internal_candidate; boundary_type = 2
                        
                if s_remaining < s_to_boundary:
                    x += s_remaining * ux; y += s_remaining * uy; z += s_remaining * uz
                    s_remaining = 0.0
                else:
                    s_taken = s_to_boundary
                    x += s_taken * ux; y += s_taken * uy; z += s_taken * uz
                    if boundary_type == 1: z = 0.0
                    elif boundary_type == 2: z = tepi_cm
                    s_remaining -= s_taken
                    
                    if boundary_type == 1:
                        cos_theta_inc = -uz
                        R_internal = calculate_fresnel_gpu(mc_n_medium, mc_n_air, cos_theta_inc)
                        local_reflectance += W * (1.0 - R_internal)
                        W *= R_internal
                        uz = -uz
                        if W < 1e-9: photon_status = mc_dead; break
                    elif boundary_type == 2:
                        current_layer_is_epi = not current_layer_is_epi
            if photon_status == mc_dead: break
            
            W *= albedo
            rnd_scatter = xoroshiro128p_uniform_float64(rng_states, thread_id)
            if abs(g_anisotropy) < 1e-6: 
                costheta = 2.0 * rnd_scatter - 1.0
            else:
                temp_g = (1.0 - g_anisotropy*g_anisotropy) / (1.0 - g_anisotropy + 2.0*g_anisotropy*rnd_scatter)
                costheta = (1.0 + g_anisotropy*g_anisotropy - temp_g*temp_g) / (2.0*g_anisotropy)
                if costheta < -1.0: costheta = -1.0
                elif costheta > 1.0: costheta = 1.0
                
            sintheta = math.sqrt(max(0.0, 1.0 - costheta*costheta))
            psi = 2.0 * mc_pi * xoroshiro128p_uniform_float64(rng_states, thread_id)
            cospsi = math.cos(psi); sinpsi = math.sin(psi)
            
            if abs(uz) > 0.99999:
                uxx = sintheta * cospsi
                uyy = sintheta * sinpsi
                uzz = costheta * (1.0 if uz >= 0.0 else -1.0)
            else:
                temp_sqrt = math.sqrt(max(0.0, 1.0 - uz*uz))
                uxx = sintheta * (ux * uz * cospsi - uy * sinpsi) / temp_sqrt + ux * costheta
                uyy = sintheta * (uy * uz * cospsi + ux * sinpsi) / temp_sqrt + uy * costheta
                uzz = -sintheta * cospsi * temp_sqrt + uz * costheta
            ux, uy, uz = uxx, uyy, uzz
            
            if W < mc_threshold:
                if xoroshiro128p_uniform_float64(rng_states, thread_id) <= mc_chance: 
                    W /= mc_chance
                else: 
                    photon_status = mc_dead
            if photon_status == mc_dead: break

    cuda.atomic.add(out_reflectance_array, wl_idx, local_reflectance)