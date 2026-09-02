import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras import models
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import ops
from keras import layers
from keras import models
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, RandomShear, RandomTranslation
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import MaxNorm
from keras.callbacks import ModelCheckpoint
from keras import utils
from keras.utils import load_img, img_to_array
from keras import optimizers
from keras.preprocessing import image
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# Mixture of Experts: 82%(90% Top-2) Accuracy Convnet Model and 82%(89% Top-2) Accuracy Vision Transformer Model
# (72, 128) images. 2-1-1 train/val/test split
# (softmax(convnet) + softmax(ViT)) / 2
# 89% accuracy.
# 94% top 2 choices accuracy.

 ###  ViT Class Definitions ###
@keras.saving.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size=8, **kwargs):
        super().__init__(**kwargs)
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

@keras.saving.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=144, projection_dim=128, **kwargs):
        super().__init__(**kwargs)
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
        config.update({"num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config
### ViT Class Definitions ###

f = open('test_set_distributions.txt', 'w')

misclassified_only=True
top_2_misclassified=0
misclassified=0
test_count=0
value_list = []
sections = {}
categories = {}
base_dir = 'dataset_72'
for x in os.listdir(base_dir):
    sections[x+'_dir'] = os.path.join(base_dir, x)
for y in os.listdir(base_dir):
    for z in os.listdir(sections[y+'_dir']):
        categories[y+'_'+z+'_dir'] = os.path.join(sections[y+'_dir'], z)
        print('total '+y+' '+z+' images: ', len(os.listdir(categories[y+'_'+z+'_dir'])))
total_categories = int(len(categories)/3)
height, width, channels = img_to_array(load_img(os.path.join(next(iter(categories.values())), next(os.scandir(next(iter(categories.values())))).name))).shape

### Models ###
model_b = load_model("fauna_conv_72.keras")
model_a=load_model(
    "fauna_vit_72.keras",
    custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder}
)

for a in os.listdir(sections["test_dir"]):
    value_list.append(a)

idx=0
for key, value in categories.items():
    if(key[0:4]=='test'):
        f.write(value+'  ---------------------------------------------------------------------------------\n')
        filepath = categories[key]
        files = iter(os.scandir(filepath))
        for x in range(0, len(os.listdir(filepath))):
            img_path = os.path.join(next(files))
            img_filename = img_path.rsplit('/', 1)[-1]
            img = image.load_img(img_path, target_size=(72, 128))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            try:
                answer_a = model_a.predict(img_tensor)
                answer_a = np.array(keras.activations.softmax(answer_a))
            except NameError:
                pass
            try:
                answer_b = model_b.predict((img_tensor/255))
                try:
                    answer_mix = (answer_a + answer_b) / 2
                except NameError:
                    answer_mix = answer_b
            except NameError:
                answer_mix = answer_a
            answer_mix = answer_mix.flatten()
            test_count+=1
            if misclassified_only==False:
                f.write(img_filename)
                f.write(' , Prediction: '+value_list[np.argmax(answer_mix)]+'\n')
                line_counter=1
                for i in range(0, len(value_list)):
                    if line_counter > 10:
                        f.write("\n")
                        f.write("  -")
                        line_counter=1
                    f.write(str(value_list[i])+': {:.0%}'.format(answer_mix[i])+', ')
                    line_counter+=1
                f.write('\n')
                f.write('\n')
            else:
                line_counter=1
                if np.argmax(answer_mix) != idx:
                    answer_2 = np.argsort(answer_mix)[-2]
                    if answer_2 != idx:
                        top_2_misclassified +=1
                    f.write(img_filename)
                    f.write(' , Prediction: '+value_list[np.argmax(answer_mix)]+'\n')
                    misclassified+=1
                    line_counter=1
                    for i in range(0, len(value_list)):
                        if line_counter > 10:
                            f.write("\n")
                            f.write("  -")
                            line_counter=1
                        f.write(str(value_list[i])+': {:.0%}'.format(answer_mix[i])+', ')
                        line_counter+=1
                    f.write('\n')
                    f.write('\n')
        idx+=1
        f.write('\n')
if misclassified_only:
    f.write("Misclassified: "+str(misclassified)+" images.\n")
    f.write(str(top_2_misclassified)+" images were not in the top 2 choices. \n")
    f.write("\n")
    print("Misclassified: "+str(misclassified)+" images.")
    print(str(top_2_misclassified)+" images were not in the top 2 choices.")
print('Tested Images: '+str(test_count))
f.close()
