import numpy as np
from skimage import color, exposure, transform
from skimage import io
import os
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras.optimizers import SGD

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import pandas as pd

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
################
def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), border_mode='same', 
                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), border_mode='same',
                            activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), border_mode='same', 
                            activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model
#model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

##############################################################
root_dir = 'GTSRB/Final_Training/Images/'
imgs = []
labels = []

#all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
#np.random.shuffle(all_img_paths)
#for img_path in all_img_paths:
#    img = preprocess_img(io.imread(img_path))
#    #print img.shape
#    label = get_class(img_path)
#    imgs.append(img)
#    labels.append(label)

#X = np.array(imgs, dtype='float32')
## Make one hot targets
#Y = np.eye(NUM_CLASSES, dtype='uint8')[labels] 

##############################################################
##############################################################
# TRAINING
##############################################################

model = cnn_model()



# let's train the model using SGD + momentum
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])
model.load_weights('model_baseline.h5')


batch_size = 32
epochs = 30

#model.fit(X, Y,
#          batch_size=batch_size,
#          epochs=epochs,
#          validation_split=0.2,
#          callbacks=[LearningRateScheduler(lr_schedule),
#                    ModelCheckpoint('model2.h5',save_best_only=True)]
#         )

###############################################################
# TESTING
###############################################################

test = pd.read_csv('GTSRB/GT-final_test.csv',sep=';')

# Load test dataset
X_test = []
y_test = []
counter = 0
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('GTSRB/Final_Test/Images/',file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)
    print counter
    counter = counter + 1
    
X_test = np.array(X_test)
y_test = np.array(y_test)

# predict and evaluate
y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/float(np.size(y_pred))
print("Test accuracy = {}".format(acc))


