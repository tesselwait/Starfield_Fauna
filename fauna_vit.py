import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

num_classes = 50
input_shape = (72, 128, 3)
BATCH_SIZE = 128

dataset="dataset_72"

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(dataset, 'train'),
    image_size=(72, 128),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(dataset, 'validation'),
    image_size=(72, 128),
    batch_size=BATCH_SIZE
)

train_ds_comb = train_ds.concatenate(val_ds)
AUTOTUNE = tf. data.AUTOTUNE
train_ds_comb = train_ds_comb.shuffle(buffer_size=500).prefetch(buffer_size=AUTOTUNE)


test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(dataset, 'test'),
    image_size=(72, 128),
    batch_size=BATCH_SIZE
)

x_list, y_list = [], []
for images, labels in train_ds_comb:
    x_list.append(images.numpy())
    y_list.append(labels.numpy())

x_train = np.concatenate(x_list, axis=0)
y_train = np.concatenate(y_list, axis=0)
y_train = np.expand_dims(y_train, axis=-1)

x2_list, y2_list = [], []
for images, labels in test_ds:
    x2_list.append(images.numpy())
    y2_list.append(labels.numpy())

x_test = np.concatenate(x2_list, axis=0)
y_test = np.concatenate(y2_list, axis=0)
y_test = np.expand_dims(y_test, axis=-1)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

learning_rate = 0.001
weight_decay = 0.00009
batch_size = BATCH_SIZE
num_epochs = 200
image_height = 72
image_width = 128
patch_size = 8
num_patches = (image_width // patch_size) * (image_height // patch_size)
projection_dim = 128
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [
    1024,
    512,
]

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_height, image_width),
        layers.RandomFlip("horizontal"),
        #layers.RandomShear(0.1, 0.1), #    shear and translation will run but won't load
        #layers.RandomTranslation(0.1, 0.1),#  in mixture of experts
        layers.RandomRotation(factor=0.0556),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    ],
    name="data_augmentation",
)
data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

@keras.saving.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")
resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_height, image_width)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_height} X {image_width}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = num_patches
plt.figure(figsize=(6, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(image_height//patch_size, image_width//patch_size, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")
plt.show()

@keras.saving.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches,})
        return config

def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.3#.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.3)#.1
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)#.5
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)#.5
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=num_epochs,
        validation_split=(1.0/3),
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    model.save('fauna_vit_72.keras')
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    return history

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
plot_history("accuracy")
