print('Loading Modules..')
import pandas as pd
import numpy as np
import os
import json
import sys
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
import keras_preprocessing.image as KPImage
from PIL import Image
import pydicom
#from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
                             TensorBoard)
from skimage.transform import downscale_local_mean
from keras.metrics import top_k_categorical_accuracy

# Load data 
print('Loading Data..')
img_info_df = pd.read_csv('./input/img_info_train.csv')
with open('params.json') as f:
    params = json.load(f)

# Encode labels
print('Encoding labels..')
class_enc = LabelEncoder()
img_info_df['class_idx'] = class_enc.fit_transform(img_info_df['word'])
oh_enc = OneHotEncoder(sparse=False)
img_info_df['class_vec'] = oh_enc.fit_transform(
    img_info_df['class_idx'].values.reshape(-1, 1)).tolist()

# Create train/validation dsets
print('Creating train/validation sets..')
train_df, val_df = train_test_split(img_info_df, test_size=params['TEST_SPLIT'],
                                    stratify=img_info_df['class_idx'])
train_df = train_df.sample(params['TRAIN_SIZE'])

class NumpyPIL:
    @staticmethod
    def open(infile):
        if infile.endswith('.npy'):
            np_arr = np.load(infile)
            inshape = np_arr.shape
            #np_arr = np_arr.reshape(inshape[0],inshape[1],1).astype('float32')
            return Image.fromarray(np_arr,'L')
        return Image.open(infile)

    fromarray = Image.fromarray


KPImage.pil_image = NumpyPIL


# Prepare datasets
print('Preparing Data..')
img_gen_params = dict(vertical_flip=True,
                      height_shift_range=0.05,
                      width_shift_range=0.02,
                      rotation_range=3.0,
                      zoom_range=0.05,
                      #preprocessing_function=preprocess_input
                      )
img_gen = KPImage.ImageDataGenerator(**img_gen_params)


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col,
                        seed=None, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              seed=seed,
                                              **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values, 0)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# For training
print('Building Training Gen')
train_gen = flow_from_dataframe(img_gen, train_df,
                                path_col='path',
                                y_col='class_vec',
                                target_size=params['IMG_SIZE'],
                                color_mode='grayscale',
                                batch_size=params['BATCH_SIZE'])
# For validation
print('Building Val Gen')
val_gen = flow_from_dataframe(img_gen, val_df,
                              path_col='path',
                              y_col='class_vec',
                              target_size=params['IMG_SIZE'],
                              color_mode='grayscale',
                              batch_size=params['BATCH_SIZE'])
# For test
print('Building Test set')
valid_X, valid_Y = next(flow_from_dataframe(img_gen, val_df,
                                            path_col='path',
                                            y_col='class_vec',
                                            target_size=params['IMG_SIZE'],
                                            color_mode='grayscale',
                                            batch_size=params['TEST_SIZE']))

# Build model
print('Building Model Structure..')
t_x, t_y = next(train_gen)

# Base
model = Sequential()
model.add(layers.Convolution2D(16,(3,3), padding='same', activation='relu',
    input_shape=t_x.shape[1:]))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size =(2,2)))
model.add(layers.Convolution2D(128, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size =(2,2)))
model.add(layers.Convolution2D(256, (3, 3), padding='same', activation= 'relu'))
model.add(layers.MaxPooling2D(pool_size =(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(t_y.shape[1], activation='softmax')) 


def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,k=3)

model.compile(optimizer=Adam(lr=params['LEARNING_RATE']),
              loss='categorical_crossentropy',
              metrics=[top_3_categorical_accuracy])

# Set up training callbacks
weight_file = "quickdraw_weights.best.hd5"
checkpoint = ModelCheckpoint(weight_file, verbose=1, save_best_only=True,
                             save_weights_only=True)
reduceLR = ReduceLROnPlateau(factor=0.8, verbose=1, cooldown=3, min_lr=0.0001)
earlystop = EarlyStopping(patience=10)
#tensorboard = TensorBoard()
callbacks = [checkpoint, reduceLR, earlystop]

# Train model
if 'train' in sys.argv:
    print('Fitting Model')
    model.fit_generator(train_gen, validation_data=(valid_X, valid_Y),
                        epochs=20, callbacks=callbacks, workers=2,
                        steps_per_epoch=params['TRAIN_SIZE']//params['BATCH_SIZE'])
else:
    print('Not fitting model')

# Save model
model.load_weights(weight_file)
model.save('full_model.h5')



# Make prediction
print('Loading Test Data..')

sub_df = pd.read_csv('img_info_test.csv')
print('read test data')
sub_gen = flow_from_dataframe(img_gen, sub_df,
                              path_col='path',
                              y_col='key_id',
                              target_size=params['IMG_SIZE'],
                              color_mode='grayscale',
                              batch_size=1,
                              shuffle=False)

steps = len(sub_df)
out_ids, out_vec = [], []
print("Making prediction..")
for _, (t_x, t_y) in zip(tqdm(range(steps)), sub_gen):
    out_vec += [model.predict(t_x)]
    out_ids += [t_y]
out_vec = np.concatenate(out_vec, 0)
out_ids = np.concatenate(out_ids, 0)

out_vec = [sorted(range(len(x)),key=lambda i:x[i])[-1:-4:-1]
        for x in out_vec]
out_vec = [" ".join(class_enc.inverse_transform(x)) for x in out_vec]
print(out_vec)


pred_df = pd.DataFrame({"key_id":out_ids,"word":out_vec})
pred_df["key_id"] = pred_df["key_id"].astype(int)
print("Saving submission..")
sub_file = 'submission.csv'
pred_df.to_csv(sub_file, index=False)
print("Submission saved as",sub_file)
