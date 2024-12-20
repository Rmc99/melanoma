import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Diretórios de dados
train_dir = "./dataset_new/Train"
test_dir = "./dataset_new/Test"

# Parâmetros
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
EPOCHS = 20

# Modificar o conjunto de classes para apenas Melanoma vs Não Melanoma
CLASS_LABELS = ['não melanoma', 'melanoma']

# Data generators
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.05,
    rescale=1./255,
    validation_split=0.2  # Proporção para validação
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Apenas normalização para o conjunto de teste

# Train, validation, and test generators (utilizando 'binary' para classificação binária)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb",
    class_mode="binary",  # Usando "binary" para Melanoma vs Não Melanoma
    subset="training"  # Para o conjunto de treinamento
)

validation_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb",
    class_mode="binary",  # Usando "binary" para Melanoma vs Não Melanoma
    subset="validation"  # Para o conjunto de validação
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb",
    class_mode="binary"  # Usando "binary" para Melanoma vs Não Melanoma
)

# Definindo o modelo CNN para classificação binária
def create_cnn_model():
    model = Sequential()

    # Definir a primeira camada (InputLayer é implícito, mas para garantir)
    model.add(InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

    # Primeiro bloco convolucional
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Segundo bloco convolucional
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Terceiro bloco convolucional
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Camadas de Flatten e Dense
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))  # Alterado para 1 unidade com 'sigmoid' para binário

    return model

# Compilando o modelo CNN
cnn_model = create_cnn_model()

cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo CNN
cnn_model.fit(
    train_generator,
    epochs=EPOCHS,  # Número de épocas ajustado
    validation_data=validation_generator
)

# Gerar previsões com o modelo CNN
cnn_predictions = cnn_model.predict(test_generator)
cnn_predictions = (cnn_predictions > 0.5).astype(int)  # Converter probabilidades para binários (0 ou 1)

# Extraindo as classes reais do test_generator
true_classes = test_generator.classes  # Rótulos reais
true_classes = true_classes.astype(int)  # Certificar-se de que sejam binários

# Calculando métricas de desempenho
accuracy = accuracy_score(true_classes, cnn_predictions)
precision = precision_score(true_classes, cnn_predictions, zero_division=0)
recall = recall_score(true_classes, cnn_predictions, zero_division=0)
f1 = f1_score(true_classes, cnn_predictions, zero_division=0)
conf_matrix = confusion_matrix(true_classes, cnn_predictions)
class_report = classification_report(true_classes, cnn_predictions, zero_division=0)

# Exibindo os resultados
print(f"Acurácia: {accuracy * 100:.2f}%")
print(f"Precisão: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")
print("Matriz de Confusão:")
print(conf_matrix)
print("Relatório de Classificação:")
print(class_report)
