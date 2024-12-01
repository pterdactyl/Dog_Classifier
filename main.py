import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

csv_path = 'C:/Users/peter/OneDrive/Documents/GitHub/Dog_Classifier/dogs.csv'
df = pd.read_csv(csv_path)

train_df = df[df['data set'] == 'train']
val_df = df[df['data set'] == 'valid']
test_df = df[df['data set'] == 'test']

# Verify splits
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Testing samples: {len(test_df)}")

image_size = (224, 224)  # Resize to match your model's input size
batch_size = 32

# Initialize ImageDataGenerators for preprocessing
datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize pixel values

# Function to create data generators from the DataFrame
def create_generator(dataframe, datagen, image_size, batch_size):
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col="filepaths",
        y_col="labels",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

# Create generators
train_generator = create_generator(train_df, datagen, image_size, batch_size)
val_generator = create_generator(val_df, datagen, image_size, batch_size)
test_generator = create_generator(test_df, datagen, image_size, batch_size)

print("Training classes:", train_generator.class_indices.keys())
print("Validation classes:", val_generator.class_indices.keys())
print("Testing classes:", test_generator.class_indices.keys())

print("Number of classes:")
print("Training:", len(train_generator.class_indices))
print("Validation:", len(val_generator.class_indices))
print("Testing:", len(test_generator.class_indices))

# Load pre-trained EfficientNetB0 and add classification layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(64, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
