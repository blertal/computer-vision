import numpy as np
from skimage import color, exposure, transform
from skimage import io
import os
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras.optimizers import SGD

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import sys
import keras

#from vgg16 import VGG16

NUM_CLASSES = 43
IMG_SIZE = 48

########################
########################
def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    #img = transform.resize(img, (224, 224))

    # roll color axis to axis 0
    #img = np.rollaxis(img,-1)

    return img

########################
########################
def get_class(img_path):
    return int(img_path.split('/')[-2])

########################
########################
def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

################

##############################################################
def vgg16_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            activation='relu', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))


    ## load the weights of the vgg16 model
    modelvgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
    modelvgg16.save('modelvgg16.h5')
    model.load_weights('modelvgg16.h5')

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    ## load the weights of the vgg16 model
#    modelvgg16 = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
#    modelvgg16.save('modelvgg16.h5')
#    model.load_weights('modelvgg16.h5')

    #model.layers.pop()
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model

##############################################################
root_dir = 'GTSRB/Final_Training/Images/'
imgs = []
labels = []

counter = 0
all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = preprocess_img(io.imread(img_path))
    #print img.shape
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)
    print counter
    counter = counter + 1

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels] 

##############################################################

model = vgg16_model()

#model = test_model()

#print 'my model ##########################'
#print model.summary()

# let's train the model using SGD + momentum
lr = 0.03
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',#sgd,
          metrics=['accuracy'])

#modelvgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

#model.get_layer('block1_conv1').set_weights(modelvgg16.get_layer('block1_conv1').get_weights())
#print model.get_layer('block1_conv1').get_weights()
#print modelvgg16.get_layer('block1_conv1').get_weights()
#sys.exit()

#model.get_layer('block1_conv1').set_weights(modelvgg16.get_layer('block1_conv1').get_weights())
#model.get_layer('block1_conv2').set_weights(modelvgg16.get_layer('block1_conv2').get_weights())
#model.get_layer('block1_pool').set_weights(modelvgg16.get_layer('block1_pool').get_weights())

#model.get_layer('block2_conv1').set_weights(modelvgg16.get_layer('block2_conv1').get_weights())
#model.get_layer('block2_conv2').set_weights(modelvgg16.get_layer('block2_conv2').get_weights())
#model.get_layer('block2_pool').set_weights(modelvgg16.get_layer('block2_pool').get_weights())

#model.get_layer('block3_conv1').set_weights(modelvgg16.get_layer('block3_conv1').get_weights())
#model.get_layer('block3_conv2').set_weights(modelvgg16.get_layer('block3_conv2').get_weights())
#model.get_layer('block3_conv3').set_weights(modelvgg16.get_layer('block3_conv3').get_weights())
#model.get_layer('block3_pool').set_weights(modelvgg16.get_layer('block3_pool').get_weights())

#model.get_layer('block4_conv1').set_weights(modelvgg16.get_layer('block4_conv1').get_weights())
#model.get_layer('block4_conv2').set_weights(modelvgg16.get_layer('block4_conv2').get_weights())
#model.get_layer('block4_conv3').set_weights(modelvgg16.get_layer('block4_conv3').get_weights())
#model.get_layer('block4_pool').set_weights(modelvgg16.get_layer('block4_pool').get_weights())

#model.get_layer('block5_conv1').set_weights(modelvgg16.get_layer('block5_conv1').get_weights())
#model.get_layer('block5_conv2').set_weights(modelvgg16.get_layer('block5_conv2').get_weights())
#model.get_layer('block5_conv3').set_weights(modelvgg16.get_layer('block5_conv3').get_weights())
#model.get_layer('block5_pool').set_weights(modelvgg16.get_layer('block5_pool').get_weights())

batch_size = 1024
epochs = 30

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('model_enhanced_1.h5',save_best_only=True)]
         )

