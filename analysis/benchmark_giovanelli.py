# --- START OF FILE analysis/benchmark_giovanelli.py ---
import numpy as np
import matplotlib.pyplot as plt
import time

def fresnel_avg(n1, n2, cos_theta1):
    cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
    if n1 == n2: return 0.0
    sin_theta1_sq = 1.0 - cos_theta1*cos_theta1
    ratio = n1 / n2
    sin_theta2_sq = (ratio*ratio) * max(sin_theta1_sq, 0.0)
    if sin_theta2_sq >= 1.0: return 1.0
    cos_theta2 = np.sqrt(1.0 - sin_theta2_sq)
    rs = ((n1*cos_theta1 - n2*cos_theta2) / (n1*cos_theta1 + n2*cos_theta2)) ** 2
    rp = ((n2*cos_theta1 - n1*cos_theta2) / (n2*cos_theta1 + n1*cos_theta2)) ** 2
    return 0.5 * (rs + rp)

def mcml_halfspace_exact(mua, mus, n_photons, g, n_medium, n_air):
    mut = mua + mus
    albedo = mus / mut if mut > 0.0 else 0.0
    R_spec = ((n_air - n_medium)/(n_air + n_medium))**2 if n_medium != n_air else 0.0
    rd_sum = 0.0

    for p in range(n_photons):
        W = 1.0 - R_spec
        x = y = z = 0.0; ux = uy = 0.0; uz = 1.0
        alive = True
        while alive:
            xi = np.random.rand()
            while xi <= 0.0: xi = np.random.rand()
            s_opt = -np.log(xi)
            while s_opt > 0.0:
                tau_b = mut * (z / -uz) if uz < 0.0 else np.inf
                if s_opt > tau_b:
                    if tau_b < np.inf:
                        step_cm = tau_b / mut
                        x += step_cm * ux; y += step_cm * uy; z = 0.0
                        s_opt -= tau_b
                        R_int = fresnel_avg(n_medium, n_air, abs(uz)) if n_medium != n_air else 0.0
                        if np.random.rand() > R_int:
                            rd_sum += W
                            alive = False
                            break
                        else:
                            uz = -uz; z = 1e-12
                            continue
                    else: break
                else:
                    step_cm = s_opt / mut
                    x += step_cm * ux; y += step_cm * uy; z += step_cm * uz
                    s_opt = 0.0; W *= albedo
                    rnd = np.random.rand()
                    if abs(g) < 1e-12: cost = 2.0 * rnd - 1.0
                    else:
                        tmp = (1.0 - g*g) / (1.0 - g + 2.0*g*rnd)
                        cost = (1.0 + g*g - tmp*tmp) / (2.0*g)
                        cost = min(max(cost, -1.0), 1.0)
                    sint = np.sqrt(max(0.0, 1.0 - cost*cost))
                    phi = 2.0 * np.pi * np.random.rand()
                    cphi, sphi = np.cos(phi), np.sin(phi)
                    if abs(uz) > 0.99999:
                        ux, uy, uz = sint*cphi, sint*sphi, cost*np.sign(uz)
                    else:
                        denom = np.sqrt(max(0.0, 1.0 - uz*uz))
                        uxx = sint*(ux*uz*cphi - uy*sphi)/denom + ux*cost
                        uyy = sint*(uy*uz*cphi + ux*sphi)/denom + uy*cost
                        uzz = -sint*cphi*denom + uz*cost
                        norm = np.sqrt(uxx*uxx + uyy*uyy + uzz*uzz)
                        ux, uy, uz = uxx/norm, uyy/norm, uzz/norm
                    if W < 0.01:
                        if np.random.rand() <= 0.1: W /= 0.1
                        else: alive = False; break
    R_d = rd_sum / n_photons
    return R_d, R_spec, R_spec + R_d

def run():
    print("--- Benchmark MCML: Caso Giovanelli ---")
    t0 = time.time()
    R_d, R_spec, R_total = mcml_halfspace_exact(10.0, 90.0, 100000, 0.0, 1.5, 1.0)
    print(f"Tempo execução: {time.time() - t0:.2f} s")

    expected_Rtot, expected_error = 0.2600, 0.0017
    plt.figure(figsize=(7, 5))
    plt.bar(['Calculado', 'Esperado'], [R_total, expected_Rtot], yerr=[0, expected_error], capsize=5, color=['skyblue', 'lightcoral'])
    plt.ylabel('Refletância Total')
    plt.title('Validação Teórica do Caso Giovanelli')
    plt.axhline(y=expected_Rtot, color='blue', linestyle='--', label=f'Esperado: {expected_Rtot:.4f}')
    plt.axhline(y=R_total, color='green', linestyle='--', label=f'Calculado: {R_total:.4f}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()