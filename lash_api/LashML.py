from lash_api.LashDatabase import DatabaseExtract
from keras import backend
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas
from keras import backend as K
from lash_api.Convertion import Convert
import json
class TensorflowML(object):
    # Here will be the instance stored.
    __FINGERPRINT_LOCATION_PATH = "C:\\Users\\User\\PycharmProjects\\lashProject\\FINGERPRINT_LOCATION.csv"
    __future = 'flid'
    __X_test_future = "x"
    __Y_test_future = "y"
    __Z_test_future = 'deck'

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(TensorflowML, cls).__new__(cls)
            cls.instance.set_database()
            cls.instance.set_convert()
            cls.instance.training()
        return cls.instance

    def get_database(self):
        return self.database

    def set_database(self):
        self.database = DatabaseExtract()

    def get_convert(self):
        return self.convert

    def set_convert(self):
        self.convert = Convert()

    def negative_predictive_value(self,y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        return tn / (tn + fn + K.epsilon())

    def matthews_correlation_coefficient(self,y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

        num = tp * tn - fp * fn
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return num / K.sqrt(den + K.epsilon())

    def rmse(self,y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def mean_squared_error(self,y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def precision(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras

    def get_model(self,column, train_features):
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.asarray(train_features).astype('float32'))

        feature = np.array(train_features[column])

        normalizer = tf.keras.layers.Normalization(input_shape=[1, ], axis=None)
        normalizer.adapt(feature)

        model = keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(1),
        ])

        model.summary()

        return model

    def training(self):

        self.FINGERPRINT_LOCATION_DATASET = self.get_database().get_panda_dataset(self.__FINGERPRINT_LOCATION_PATH)


        self.train_dataset =  self.FINGERPRINT_LOCATION_DATASET.sample(frac=0.9, random_state=0)
        self.test_dataset =  self.FINGERPRINT_LOCATION_DATASET.drop(self.train_dataset.index)


        X_train_features = self.train_dataset.copy()
        self.X_test_features = self.test_dataset.copy()

        Y_train_features = self.train_dataset.copy()
        self.Y_test_features = self.test_dataset.copy()

        Z_train_features = self.train_dataset.copy()
        self.z_test_features = self.test_dataset.copy()

        Y_train_labels = self.train_dataset.pop('y')


        X_train_labels = self.train_dataset.pop('x')


        Z_train_labels = self.train_dataset.pop('deck')


        future = 'flid'
        X_test_future = "x"
        Y_test_future = "y"
        Z_test_future = 'deck'

        self.__X_model = self.get_model(future, X_train_features)
        self.__Y_model = self.get_model(future, Y_train_features)
        self.__Z_model = self.get_model(future, Z_train_features)
        # loss and optimizer
        loss = keras.losses.mean_squared_error  # MeanSquaredError
        optim = keras.optimizers.SGD(lr=0.8)
        optim2 = keras.optimizers.SGD(lr=0.8)
        optim3 = keras.optimizers.SGD(lr=0.8)
        self.__X_model.compile(optimizer=optim, loss=loss, metrics=[
            self.rmse,
            self.mean_squared_error,
            self.negative_predictive_value,
            self.matthews_correlation_coefficient,
            self.precision
        ])
        self.__Y_model.compile(optimizer=optim2, loss=loss, metrics=[
            self.rmse,
            self.mean_squared_error,
            self.negative_predictive_value,
            self.matthews_correlation_coefficient,
            self.precision
        ])
        self.__Z_model.compile(optimizer=optim3, loss=loss, metrics=[
            self.rmse,
            self.mean_squared_error,
            self.negative_predictive_value,
            self.matthews_correlation_coefficient,
            self.precision
        ])

        self.__X_model.fit(X_train_features[future], X_train_labels, epochs=300, verbose=1, validation_split=0.8)
        self.__Y_model.fit(Y_train_features[future], Y_train_labels, epochs=300, verbose=1, validation_split=0.8)
        self.__Z_model.fit(Z_train_features[future], Z_train_labels, epochs=300, verbose=1, validation_split=0.8)

    def get_dist(self,distances):
        return distances.get('Distance')

    def predict(self,oids):
        self.get_database().set_oid(oids)
        flid, x_proto, y_proto, z_proto = self.get_database().extraction_flid()
        x_zeros = []
        y_zeros = []
        z_zeros = []
        x = x_proto.copy()
        y = y_proto.copy()
        z = z_proto.copy()
        for i in range((len(self.test_dataset) // len(x) - 1)):
            x = x + x
        for i in range((len(self.test_dataset) // len(y) - 1)):
            y = y + y
        for i in range((len(self.test_dataset) // len(z) - 1)):
            z = z + z
        for i in range(len(self.test_dataset) - len(x)):
            x_zeros.insert(0, tf.reduce_min(x).numpy())
        for i in range(len(self.test_dataset) - len(y)):
            y_zeros.insert(0, tf.reduce_min(y).numpy())
        for i in range(len(self.test_dataset) - len(z)):
            z_zeros.insert(0, tf.reduce_min(z).numpy())

        self.Y_test_labels = pandas.Series(y + y_zeros)
        self.X_test_labels = pandas.Series(x + x_zeros)
        self.Z_test_labels = pandas.Series(z + z_zeros)

        self.__X_model.evaluate(self.X_test_features[self.__X_test_future], self.X_test_labels, verbose=1)
        self.__Y_model.evaluate(self.Y_test_features[self.__Y_test_future], self.Y_test_labels, verbose=1)
        self.__Z_model.evaluate(self.z_test_features[self.__Z_test_future], self.Z_test_labels, verbose=1)

        X_prediction = self.__X_model.predict(tf.constant(flid))
        Y_prediction = self.__Y_model.predict(tf.constant(flid))
        Z_prediction = self.__Z_model.predict(tf.constant(flid)).astype(int)

        dist = self.get_convert().get_lat_long_to_meters(flid, X_prediction, Y_prediction, x_proto, y_proto)
        result = []
        self.FINGERPRINT_LOCATION_DATASET = self.FINGERPRINT_LOCATION_DATASET[
            self.FINGERPRINT_LOCATION_DATASET['flid'].isin(flid)].drop('time', axis=1).drop('modelid', axis=1)
        self.FINGERPRINT_LOCATION_DATASET.insert(3, "Distance", dist[0][0], True)
        for index, row in self.FINGERPRINT_LOCATION_DATASET.iterrows():
            val = {
                "flid": row['flid'],
                "x": row['x'],
                "y": row['y'],
                "deck": row['deck'],
                "Distance": row['Distance'],
            }
            result.insert(i, val)
        result.sort(key=self.get_dist)
        return result


