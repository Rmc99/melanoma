import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro, f_oneway

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

# Calculando a diferença entre as médias das métricas
differences = cnn_avg_metrics - ensemble_avg_metrics

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
t_stat_ttest, p_value_ttest = ttest_rel(cnn_avg_metrics, ensemble_avg_metrics)
print("\nTeste t pareado:")
print(f"Estatística t: {t_stat_ttest:.4f}")
print(f"Valor-p: {p_value_ttest:.4f}")

# Teste de Wilcoxon
if p_value_shapiro <= 0.05:  # Realizar Wilcoxon se não seguir normalidade
    stat_wilcoxon, p_value_wilcoxon = wilcoxon(differences)
    print("\nTeste de Wilcoxon:")
    print(f"Estatística W: {stat_wilcoxon:.4f}")
    print(f"Valor-p: {p_value_wilcoxon:.4f}")

# Teste ANOVA (para múltiplos grupos)
print("\nTeste ANOVA (mais de dois grupos):")
f_stat_anova, p_value_anova = f_oneway(cnn_metrics.flatten(), ensemble_metrics.flatten())
print(f"Estatística F: {f_stat_anova:.4f}")
print(f"Valor-p: {p_value_anova:.4f}")
if p_value_anova > 0.05:
    print("Não há diferença estatisticamente significativa entre os grupos.")
else:
    print("Há diferença estatisticamente significativa entre os grupos.")
