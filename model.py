from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
import PIL

classifier = Sequential()


classifier.add(Conv2D(64,(3,3),input_shape=(150,150,3), activation='relu'))

# MaxPooling
classifier.add(MaxPooling2D(pool_size = (2,2)))


classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(132,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


classifier.add(Conv2D(128,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(132,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())


classifier.add(Dense(units = 132, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.summary()

train_data_dir=r'F:\Chotu\docspot\new data\Gender-Detection-master\gender_dataset_face'

batch_size=32

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data


label_map = (train_generator.class_indices)
print(label_map)


history=classifier.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        verbose=1)


classifier.save("model.h5")