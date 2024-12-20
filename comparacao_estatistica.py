import numpy as np
from scipy.stats import ttest_rel

# Simulando execuções múltiplas para CNN e Ensemble
cnn_metrics = np.array([
    [81.25, 85.71, 75.00, 80.00],  # Execução 1
    [84.38, 100.00, 68.75, 81.48],  # Execução 2
    [81.25, 85.71, 75.00, 80.00],  # Execução 3
])
ensemble_metrics = np.array([
    [90.62, 93.33, 87.50, 91.00],  # Execução 1
    [87.50, 92.86, 81.25, 86.67],  # Execução 2
    [81.25, 100, 62.50, 76.92],  # Execução 3
])

# Calculando a média das execuções para cada métrica
cnn_avg_metrics = cnn_metrics.mean(axis=0)
ensemble_avg_metrics = ensemble_metrics.mean(axis=0)

# Teste t pareado
t_stat_ttest, p_value_ttest = ttest_rel(cnn_avg_metrics, ensemble_avg_metrics)
print("Teste t pareado:")
print(f"Estatística t: {t_stat_ttest:.4f}")
print(f"Valor-p: {p_value_ttest:.4f}")

if p_value_ttest < 0.05:
    print("Há diferença estatisticamente significativa entre as médias dos grupos.")
else:
    print("Não há diferença estatisticamente significativa entre as médias dos grupos.")
