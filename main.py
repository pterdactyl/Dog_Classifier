import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os

# Load the dataset
df = pd.read_csv('dogs.csv')

# Split dataset into train, validation, and test sets
train_df = df[df['data set'] == 'train']
val_df = df[df['data set'] == 'valid']
test_df = df[df['data set'] == 'test']

# Verify splits
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Testing samples: {len(test_df)}")

image_size = (224, 224)  # Reduced image size for faster training
batch_size = 16  # Reduced batch size

# Initialize ImageDataGenerators for preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,  # Reduced augmentation range
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

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

# Load pre-trained EfficientNetB0 and add classification layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tune by unfreezing top layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[lr_scheduler]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
