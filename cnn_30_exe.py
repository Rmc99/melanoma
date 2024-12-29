import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Diretórios de dados
train_dir = "./dataset_new/Train"
test_dir = "./dataset_new/Test"

# Parâmetros
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
EPOCHS = 20
NUM_EXECUTIONS = 30

# Classes para classificação binária
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

# Train, validation, and test generators
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

# Definição do modelo CNN
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

# Lista para armazenar os resultados
results = []

# Loop de execuções
for execution in range(NUM_EXECUTIONS):
    print(f"Execução {execution + 1}/{NUM_EXECUTIONS}")

    # Criar e compilar o modelo
    cnn_model = create_cnn_model()
    cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Treinamento do modelo
    cnn_model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=0  # Suprimir logs para execuções múltiplas
    )

    # Previsões no conjunto de teste
    cnn_predictions = cnn_model.predict(test_generator)
    cnn_predictions = (cnn_predictions > 0.5).astype(int)

    # Rótulos reais
    true_classes = test_generator.classes

    # Cálculo das métricas
    accuracy = accuracy_score(true_classes, cnn_predictions)
    precision = precision_score(true_classes, cnn_predictions, zero_division=0)
    recall = recall_score(true_classes, cnn_predictions, zero_division=0)
    f1 = f1_score(true_classes, cnn_predictions, zero_division=0)

    # Armazenar os resultados
    results.append({
        "Execução": execution + 1,
        "Acurácia": accuracy,
        "Precisão": precision,
        "Recall": recall,
        "F1-Score": f1
    })

# Salvar os resultados em um arquivo CSV
results_df = pd.DataFrame(results)
results_df.to_csv("cnn_results.csv", index=False)

print("Resultados salvos em 'cnn_results.csv'")
