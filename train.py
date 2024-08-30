import tensorflow as tf
from keras import layers, models, optimizers, callbacks, mixed_precision, preprocessing, applications

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Training on CPU is orders of magnitude slower, idk why tensorflow even allows it by default
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if len(gpus) == 0:
    raise RuntimeError("GPU not found")

# Parameters
BATCH_SIZE = 64
IMG_SIZE = 224
DATASET_PATH = './dataset'

# Data preprocessing
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Data augmentation
    # We can't use flip_left_right or flip_up_down because the orientation is important
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    
    # Random rotation, use the rotation as the label
    rotation = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, rotation)
    
    return image, rotation

# Data pipeline
base_ds = tf.data.Dataset.list_files(DATASET_PATH + '/*', shuffle=True)
base_ds = base_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
base_ds = base_ds.batch(BATCH_SIZE)
base_ds = base_ds.prefetch(tf.data.AUTOTUNE)

train_size = int(0.8 * len(base_ds))
train_ds = base_ds.take(train_size)
val_ds = base_ds.skip(train_size)

# Model setup
base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

# Callbacks for training
reduce_lr = callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('models/best_model.keras', save_best_only=True)

# Training
model.compile(optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

model.save('models/initial_model.keras')

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

model.save('models/final_model.keras')
