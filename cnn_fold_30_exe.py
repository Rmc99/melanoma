import csv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Diretórios de dados
train_dir = "./dataset_new/Train"
test_dir = "./dataset_new/Test"

# Parâmetros
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
EPOCHS = 20
K = 5  # Número de folds
EXECUTIONS = 30  # Número de execuções
OUTPUT_CSV = "cnn_average_results.csv"

# Modificar o conjunto de classes para apenas Melanoma vs Não Melanoma
CLASS_LABELS = ['não melanoma', 'melanoma']

# Data generators
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.05,
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb",
    class_mode="binary",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb",
    class_mode="binary",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb",
    class_mode="binary"
)

# Definindo o modelo CNN para classificação binária
def create_cnn_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model

# Extração de dados para k-fold
data = []
labels = []
for i in range(len(train_generator)):
    batch_data, batch_labels = train_generator[i]
    data.append(batch_data)
    labels.append(batch_labels)
data = np.vstack(data)
labels = np.hstack(labels)

# Inicializar lista para salvar resultados
average_results = []

# Executar o treinamento e avaliação várias vezes
for execution in range(EXECUTIONS):
    print(f"Execução {execution + 1}/{EXECUTIONS}")
    kf = KFold(n_splits=K, shuffle=True, random_state=execution)

    execution_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1_Score': []}

    # Loop para k-fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}/{K}")

        # Divisão dos dados
        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # Criar e treinar o modelo CNN
        cnn_model = create_cnn_model()
        cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        cnn_model.fit(train_data, train_labels, epochs=EPOCHS, validation_data=(val_data, val_labels), verbose=0)

        # Previsões CNN
        cnn_predictions = cnn_model.predict(test_generator)
        cnn_predictions = (cnn_predictions > 0.5).astype(int)

        # Métricas
        true_classes = test_generator.classes
        acc = accuracy_score(true_classes, cnn_predictions)
        precision = precision_score(true_classes, cnn_predictions, zero_division=0)
        recall = recall_score(true_classes, cnn_predictions, zero_division=0)
        f1 = f1_score(true_classes, cnn_predictions, zero_division=0)

        print(f"Fold {fold + 1} concluído: Acurácia={acc:.2f}, Precisão={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}")

        # Salvar métricas para a execução
        execution_metrics['Accuracy'].append(acc)
        execution_metrics['Precision'].append(precision)
        execution_metrics['Recall'].append(recall)
        execution_metrics['F1_Score'].append(f1)

    # Calcular médias por execução
    avg_accuracy = np.mean(execution_metrics['Accuracy'])
    avg_precision = np.mean(execution_metrics['Precision'])
    avg_recall = np.mean(execution_metrics['Recall'])
    avg_f1 = np.mean(execution_metrics['F1_Score'])

    print(f"Médias da execução {execution + 1}: Acurácia={avg_accuracy:.2f}, Precisão={avg_precision:.2f}, Recall={avg_recall:.2f}, F1-Score={avg_f1:.2f}")

    # Salvar médias na lista de resultados
    average_results.append({
        'Execution': execution + 1,
        'Avg_Accuracy': avg_accuracy,
        'Avg_Precision': avg_precision,
        'Avg_Recall': avg_recall,
        'Avg_F1_Score': avg_f1
    })

# Salvar resultados em um arquivo CSV
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Execution', 'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_F1_Score'])
    writer.writeheader()
    writer.writerows(average_results)

print("Treinamento concluído e médias salvas em cnn_average_results.csv.")
