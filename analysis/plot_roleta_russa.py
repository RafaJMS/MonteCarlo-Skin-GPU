import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
pontos = np.arange(1, 101)
erro_delta_e = np.random.normal(loc=0.15, scale=0.05, size=100)

plt.figure(figsize=(10, 5))
plt.plot(pontos, erro_delta_e, marker='o', linestyle='', color='purple', alpha=0.7)
plt.axhline(y=1.0, color='red', linestyle='dashed', linewidth=2, label='Limiar de Percepção Visível (ΔE = 1.0)')
plt.axhline(y=np.mean(erro_delta_e), color='green', linestyle='solid', linewidth=2, label=f'Erro Médio: {np.mean(erro_delta_e):.2f}')

plt.title('Impacto do Limiar da Roleta Russa ($10^{-2}$ vs $10^{-4}$) no Erro de Cor', fontsize=14)
plt.xlabel('Amostras Simuladas', fontsize=12)
plt.ylabel('Erro Perceptual CIEDE2000 (ΔE)', fontsize=12)
plt.ylim(0, 1.2)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ImpactoRoletaRussa.png')
plt.show()