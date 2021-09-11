import warnings
warnings.filterwarnings('ignore')

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from keras import regularizers
from keras import __version__
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


IM_WIDTH, IM_HEIGHT = 299, 299 #InceptionResNetV2 input size
FC_SIZE = 1024                # fully connected layer nodes
NB_IV3_LAYERS_TO_FREEZE = 172  # froze layers
train_dir = r'E:\dataset\train' 
val_dir = r'E:\dataset\val' 
nb_classes= 50
nb_epoch = 24
batch_size = 32
  

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = layers.BatchNormalization()(x)#add BatchNormalization layer
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)#add dropout layer
    x = Dense(FC_SIZE, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x) #new FC layer, random init and add regularizer layers
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_finetune(model):
    """
    Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training and validation loss')
    plt.legend(loc='upper right')
    plt.show()

nb_train_samples = get_nb_files(train_dir) 
nb_classes = len(glob.glob(train_dir + "/*")) 
nb_val_samples = get_nb_files(val_dir)       
nb_epoch = int(nb_epoch)                
batch_size = int(batch_size)

#　ImageDataGenerator
train_datagen =  ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rotation_range=30,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  rotation_range=30,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

# train and test data generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IM_WIDTH, IM_HEIGHT),
                                                    batch_size=batch_size,class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(val_dir,
                                                        target_size=(IM_WIDTH, IM_HEIGHT),
                                                        batch_size=batch_size,class_mode='categorical')

# setup model
base_model = InceptionResNetV2(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
model = add_new_last_layer(base_model, nb_classes)


setup_to_finetune(model)
#tensorboard
log_dir = './model'
tensorBoard = TensorBoard(
    log_dir=log_dir,
    write_graph=True,
    write_images=True)


history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    #verbose = 2,
    callbacks=[tensorBoard],
    validation_data=validation_generator,
    validation_steps=nb_val_samples//batch_size+1,
    class_weight='auto')

# save model
# model.save('./model/inception_model.h5')
plot_training(history_ft)
