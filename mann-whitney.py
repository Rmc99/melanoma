import numpy as np
import scipy.stats as stats
import pandas as pd

# Resultados dos modelos (já fornecidos)
ensemble_results = {
    'Accuracy': [0.8053191489361702,
                 0.7936170212765958,
                 0.7946808510638298,
                 0.8223404255319149,
                 0.7936170212765957,
                 0.7861702127659574,
                 0.8095744680851065,
                 0.801063829787234,
                 0.8074468085106383,
                 0.7882978723404256,
                 0.7861702127659574,
                 0.8191489361702129,
                 0.8127659574468085,
                 0.7872340425531915,
                 0.7882978723404255,
                 0.824468085106383,
                 0.798936170212766,
                 0.7978723404255319,
                 0.8074468085106383,
                 0.7914893617021277,
                 0.7808510638297872,
                 0.8265957446808511,
                 0.7840425531914894,
                 0.7829787234042553,
                 0.8148936170212766,
                 0.7957446808510638,
                 0.7872340425531915,
                 0.7776595744680851,
                 0.827659574468085,
                 0.8085106382978724,
    ]
}

cnn_results = {
    'Accuracy': [0.8127659574468085,
                 0.7893617021276595,
                 0.7978723404255319,
                 0.7872340425531915,
                 0.778723404255319,
                 0.8148936170212766,
                 0.824468085106383,
                 0.8106382978723404,
                 0.8202127659574469,
                 0.8063829787234041,
                 0.8191489361702127,
                 0.8202127659574469,
                 0.8138297872340425,
                 0.8095744680851062,
                 0.7819148936170212,
                 0.8127659574468084,
                 0.7840425531914894,
                 0.8212765957446809,
                 0.8063829787234044,
                 0.8265957446808511,
                 0.8085106382978724,
                 0.803191489361702,
                 0.7872340425531915,
                 0.7861702127659573,
                 0.8138297872340425,
                 0.7904255319148936,
                 0.8223404255319149,
                 0.8085106382978724,
                 0.8127659574468085,
                 0.798936170212766,
    ]
}

# Realizando o teste de Mann-Whitney para a acurácia dos modelos
stat, p_value = stats.mannwhitneyu(ensemble_results['Accuracy'], cnn_results['Accuracy'], alternative='two-sided')

# Exibindo o resultado do teste
print(f"Mann-Whitney U test result for Accuracy:")
print(f"U-statistic: {stat}")
print(f"p-value: {p_value}")
