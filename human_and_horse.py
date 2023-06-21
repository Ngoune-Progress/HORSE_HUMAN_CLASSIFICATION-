import  tensorflow as tf
from tensorflow.keras.applications import  MobileNet

mbl = MobileNet(include_top=False)

for layer in mbl.layers:
    layer.trainable = False

train ='/Users/ngounepeetprogress/Desktop/datasets/CatVSDog_classifitier/train'
test ='/Users/ngounepeetprogress/Desktop/datasets/CatVSDog_classifitier/test'

training_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)
testing_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training = training_gen.flow_from_directory(train,target_size=(150,150),batch_size=32,class_mode='binary')
testing = testing_gen.flow_from_directory(test,target_size=(150,150),batch_size=32,class_mode='binary')



inputs = tf.keras.Input(shape=(150,150,3))
x = mbl(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(inputs,output)


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


class Stop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy')>0.9):
            self.model.stop_training = True

check = tf.keras.callbacks.ModelCheckpoint('check',save_best_only=True)
tf.keras.layers.MaxPooling2D

model.fit(training,steps_per_epoch=100,validation_data=testing,validation_steps=100,epochs=10,batch_size=32,callbacks=[Stop(),check])

lm = tf.keras.models.load_model('check')

lm.evaluate(testing)

lm.save('Models/horse_human.h5')