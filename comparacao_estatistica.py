import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro

# Definindo as métricas dos dois modelos (CNN e Ensemble)
cnn_metrics = np.array([81.25, 85.71, 75.00, 80.00])
ensemble_metrics = np.array([90.62, 93.33, 87.50, 91.00])

# Calculando a diferença entre as métricas
differences = cnn_metrics - ensemble_metrics

# Teste de normalidade com Shapiro-Wilk
test_stat_shapiro, p_value_shapiro = shapiro(differences)
print("Teste de Shapiro-Wilk:")
print(f"Estatística do teste: {test_stat_shapiro:.4f}")
print(f"Valor-p: {p_value_shapiro:.4f}")
if p_value_shapiro > 0.05:
    print("As diferenças seguem uma distribuição normal. Use o Teste t pareado.")
else:
    print("As diferenças não seguem uma distribuição normal. Use o Teste de Wilcoxon.")

# Teste t pareado
t_stat_ttest, p_value_ttest = ttest_rel(cnn_metrics, ensemble_metrics)
print("\nTeste t pareado:")
print(f"Estatística t: {t_stat_ttest:.4f}")
print(f"Valor-p: {p_value_ttest:.4f}")