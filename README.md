# 🧬 Simulador Bioplausível de Refletância da Pele Humana (GPU/CUDA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Numba-CUDA-green.svg)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

Este repositório contém um simulador estocástico avançado baseado no método **Monte Carlo para Transporte de Luz em Tecidos Multicamadas (MCML)**. Projetado para aplicações em Computação Gráfica e Física Médica, o modelo simula a interação de fótons com a epiderme e a derme, gerando espectros de refletância e convertendo-os em cores no espaço sRGB de forma fisicamente correta.

O projeto conta com **aceleração massivamente paralela em GPU via Numba CUDA**, reduzindo o tempo de cálculo de espectros complexos de horas para minutos, com um sistema de *fallback* automático para CPU multi-core.

---

## ✨ Principais Funcionalidades

*   **Física Óptica Rigorosa:** Implementa equações de Fresnel, espalhamento anisotrópico de Henyey-Greenstein, e modelo de absorção multicamada.
*   **Parâmetros Biológicos Reais:** Modulação baseada na fração de melanina (Eumelanina/Feomelanina) e fração volumétrica de sangue (Oxi/Desoxi-hemoglobina).
*   **Aceleração HPC (High-Performance Computing):** Kernel CUDA personalizado processando o espectro completo (380nm a 780nm) simultaneamente em arquitetura paralela.
*   **Geração de LUTs (Look-Up Tables):** Exportação nativa de dados para `.csv`, imagens 2D (`.png`) e matrizes tridimensionais (`.cube`) prontas para softwares de renderização e color grading.
*   **Colorimetria CIE 1931:** Integração matemática precisa do espectro sob o Iluminante Padrão D65 para o espaço de cor perceptivo humano.

---

## 📊 Validações Científicas e Resultados

Este projeto foi rigorosamente validado através de 8 testes científicos (os scripts reproduzíveis estão na pasta `/analysis`).

1. **Validação da Física de Base (Caso Giovanelli):** O modelo bate perfeitamente com os valores teóricos analíticos para refletância em meio semi-infinito.
2. **Convergência Estatística:** Demonstração da Lei dos Grandes Números, com a variância da refletância difusa caindo linearmente (escala log-log) estabilizando o ruído de Monte Carlo com $N=10^5$ fótons.
3. **Benchmarking HPC (CPU vs GPU):** Speedup exponencial com a placa de vídeo processando até 81 comprimentos de onda por pixel simultaneamente contra o processamento multithread de CPU.
4. **Sensibilidade de Parâmetros (Correlação de Pearson):** Matriz de correlação comprovando que a fração de melanina ($C_m$) dita a luminância global, enquanto a hemoglobina ($C_h$) modula o balanço verde/vermelho.
5. **Assinatura Espectral (Curva em "W"):** Validação da captura dos picos de absorção dupla da Oxi-hemoglobina exatos em ~542nm e ~577nm.
6. **Quantificação de Metamerismo:** Histograma mapeando o "problema inverso" da pele, provando matematicamente quantas biologias distintas resultam na exata mesma cor RGB sob a luz D65.
7. **Erro de Discretização Perceptual (CIEDE2000):** Teste de *Ground Truth* com 1.000 amostras aleatórias comprovando que a interpolação trilinear da LUT possui um **Erro Médio ($\Delta E_{00}$) de apenas 0.55**. Como $\Delta E < 1.0$, as aproximações da LUT são visualmente indistinguíveis da simulação direta para o olho humano.
8. **Escala Visual (Fitzpatrick):** Matrizes 2D visualizando o gradiente natural de tons de pele derivados puramente da biologia.

---

## 📂 Arquitetura do Projeto

O projeto foi construído sob rigorosos padrões de engenharia de software para garantir modularidade e reutilização de código:

```text
MeuProjetoPele/
│
├── data/
│   └── BiologicalParameters/       # Espectros base (CIE CMFs, Melanina, Hemoglobina, etc.)
│
├── core/                           # Módulos centrais independentes
│   ├── data_loader.py              # Parsers e funções de interpolação
│   ├── optics.py                   # Cálculos de absorção e espalhamento (mua, mus)
│   ├── color_science.py            # Integração espectral e conversão de cor XYZ/sRGB
│   ├── monte_carlo_cpu.py          # Motor de simulação CPU (Numba JIT)
│   └── monte_carlo_gpu.py          # Motor de simulação GPU (Numba CUDA Batch)
│
├── main_generator.py               # Script principal (Painel de Controle e pipeline)
│
└── analysis/                       # Suíte de Validação Científica
    ├── benchmark_giovanelli.py
    ├── plot_correlation.py
    ├── plot_metamerism.py
    ├── plot_w_curve.py
    └── validar_discretizacao_lut.py
