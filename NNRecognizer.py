import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.models import model_from_json


class NNRecognizer:

    def __init__(self):
        self.model = self.load_model()

    def create_and_fit(self):
        train = pd.read_csv('train.csv')

        y = train['label']
        X = train.drop(['label'],axis = 1)

        X = X.values
        y = to_categorical(y)
        X = X.reshape(42000,28,28,1)
        X = X/255

        model = Sequential()

        model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(10,activation='softmax'))

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

        model.fit(X,y,epochs=3)
        self.save_model(model)
        self.model = model

    def save_model(self, model):
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self):
        try:
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            print("Loaded model from disk")
            return loaded_model
        except IOError:
            print("File not accessible")
            print("Create and save new model")
            self.create_and_fit()

    def predict(self,data):
        pred = self.model.predict(data)
        return pred.argmax(axis=1)
