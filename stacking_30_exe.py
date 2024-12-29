import csv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Diretórios de dados
train_dir = "./dataset_new/Train"
test_dir = "./dataset_new/Test"

# Parâmetros
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
EPOCHS = 20
NUM_EXECUTIONS = 30  # Número de execuções

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

# CSV para armazenar resultados
csv_file = "stacking_results.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Execution", "Accuracy", "Precision", "Recall", "F1-Score"])

# Loop para múltiplas execuções
for execution in range(1, NUM_EXECUTIONS + 1):
    print(f"Execução {execution}/{NUM_EXECUTIONS}")

    # Criar e treinar o modelo CNN
    cnn_model = create_cnn_model()
    cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, verbose=0)

    # Previsões CNN
    cnn_predictions = cnn_model.predict(test_generator)
    cnn_predictions = (cnn_predictions > 0.5).astype(int)

    # Extraindo características para o RandomForest
    cnn_feature_extractor = Sequential(cnn_model.layers[:-1])
    train_features = cnn_feature_extractor.predict(train_generator)
    train_features = train_features.reshape(train_features.shape[0], -1)
    test_features = cnn_feature_extractor.predict(test_generator)
    test_features = test_features.reshape(test_features.shape[0], -1)

    # RandomForest
    rf_classifier = RandomForestClassifier(n_estimators=200)
    rf_classifier.fit(train_features, train_generator.classes)
    rf_predictions = rf_classifier.predict(test_features)

    # Meta-modelo Logistic Regression
    meta_model = LogisticRegression()
    true_classes = test_generator.classes
    meta_model.fit(np.column_stack((cnn_predictions.flatten(), rf_predictions)), true_classes)

    # Previsões finais
    final_predictions = meta_model.predict(np.column_stack((cnn_predictions.flatten(), rf_predictions)))

    # Métricas
    stacking_acc = accuracy_score(true_classes, final_predictions)
    precision = precision_score(true_classes, final_predictions, zero_division=0)
    recall = recall_score(true_classes, final_predictions, zero_division=0)
    f1 = f1_score(true_classes, final_predictions, zero_division=0)

    # Armazenar resultados no arquivo CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([execution, stacking_acc, precision, recall, f1])

    print(f"Execução {execution} concluída: Acurácia={stacking_acc:.2f}, Precisão={precision:.2f}, Recall={recall:.2f}, F1-Score={f1:.2f}")

print(f"Resultados salvos em {csv_file}.")

