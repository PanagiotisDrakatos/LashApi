from functools import reduce
from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os

class MovieLensModel(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate. Under the hood,
    # these are still plain Keras Models.

    def __init__(
            self,
            user_model: tf.keras.Model,
            movie_model: tf.keras.Model,
            task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.movie_model = movie_model

        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.

        user_embeddings = self.user_model(features["id"])
        movie_embeddings = self.movie_model(features["flid"])

        return self.task(user_embeddings, movie_embeddings)


class Factorization(object):
    __file_path = os.getcwd()+"\\FINGERPRINT_OBJECT.csv"
    __file_path2 = os.getcwd()+"\\FINGERPRINT.csv"

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Factorization, cls).__new__(cls)
            cls.instance.setup()
        return cls.instance

    def get_rating(self):
        return self.rating

    def set_rating(self):
        self.rating = self.ratings(self.__file_path2)

    def get_rating_long(self):
        return self.rating_long

    def set_rating_long(self):
        self.rating_long = self.ratings2(self.__file_path2)

    def get_fingerprints(self):
        return self.fingerprints

    def set_fingerprints(self):
        self.fingerprints = self.fingerprints(self.__file_path)

    def fingerprints(self,file_path):
        fingerprints = pd.read_csv(file_path, index_col=None, encoding='latin-1')
        fingerprints = fingerprints[['flid', 'oid']].reset_index(drop=True)
        fingerprints = fingerprints.drop_duplicates('oid')
        fingerprints['counter'] = range(len(fingerprints))
        fingerprints['flid'] = fingerprints['flid'].astype(str)
        fingerprints['oid'] = fingerprints['oid'].astype(str)
        return fingerprints

    def ratings(self,file_path):
        ratings = pd.read_csv(file_path, index_col=None, encoding='latin-1')
        # ratings = ratings[["flid", "oid"]].head(10).rename(columns={"oid": "id"})
        ratings = ratings[["flid", "oid"]].rename(columns={"oid": "id"})
        # ratings['id'] = ratings['id'].astype('|S80').map(str)
        ratings['flid'] = ratings['flid'].astype(str)
        # ratings.astype({'id': 'string', 'flid': 'string'}).dtypes
        return ratings

    def ratings2(self,file_path):
        ratings2 = pd.read_csv(file_path, index_col=None, encoding='latin-1')
        ratings2 = ratings2[["flid", 'time', "oid", "x", "y", "deck"]]
        ratings2.drop(ratings2.loc[ratings2['time'] == 1658826977].index, inplace=True)
        return ratings2



    def setup(self):
        self.set_fingerprints()
        self.set_rating()
        self.set_rating_long();

        self.fingerprints_frame = tf.data.Dataset.from_tensor_slices(dict(self.get_fingerprints()))
        self.ratings = tf.data.Dataset.from_tensor_slices(dict(self.rating))
        self.ratings =  self.ratings.map(lambda x: {
            "flid": x["flid"],
            "id": x["id"]
        })
        self.fingerprints_frame = self.fingerprints_frame.map(lambda x: x["oid"])

        ids_vocabulary = tf.keras.layers.IntegerLookup(mask_token=None)
        ids_vocabulary.adapt(self.ratings.map(lambda x: x["id"]))

        movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        movie_titles_vocabulary.adapt(self.fingerprints_frame)
        # Define user and movie models.
        user_model = tf.keras.Sequential([
            ids_vocabulary,
            tf.keras.layers.Embedding(ids_vocabulary.vocabulary_size(), 64)
        ])
        movie_model = tf.keras.Sequential([
            movie_titles_vocabulary,
            tf.keras.layers.Embedding(movie_titles_vocabulary.vocabulary_size(), 64)
        ])

        # Define your objectives.
        task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            self.fingerprints_frame.batch(128).map(movie_model)
        )
        )

        # Create a retrieval model.
        self.model = MovieLensModel(user_model, movie_model, task)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(0.8))
        self.model.fit(self.ratings.batch(1024), epochs=10)


    def reccomend(self, oids):
        result_frame = []
        for index, item in enumerate(oids):
            frequency = []
            K = 0

            index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_model, k=122)
            index.index_from_dataset(self.fingerprints_frame.batch(128).map(lambda title: (title, self.model.movie_model(title))))
            # Use brute-force search to set up retrieval using the trained representations.
            # Get some recommendations.
            _, titles = index(np.array([item]))
            titles = titles.numpy()
            for j in range(titles.size):
                frequency.insert(K, titles.item(j))
                K = K + 1

            recommended = ({
                'Recommendation': frequency,
            })
            frame = pd.DataFrame(recommended, columns=['Recommendation'])
            result_frame.append(frame)

        updated = []
        for i in range(len(result_frame)):
            result_frame[i] = result_frame[i].rename(columns={"Recommendation": "flid"})
            result_frame[i] = result_frame[i]['flid'].astype(np.int64)
            rating_long_res = self.get_rating_long().merge(result_frame[i], on='flid', how='left', indicator=True)
            final_res = rating_long_res[rating_long_res['oid'].isin(oids)]
            final_res = final_res.groupby(['flid'])['oid'].nunique().to_frame('Frequency').reset_index().sort_values(
                'Frequency', ascending=False)
            updated.append(final_res)
        result_frame = reduce(lambda left, right: pd.merge(left, right, on=['flid'], how='inner'), updated).fillna('none')
        result_frame = result_frame.loc[:, ~result_frame.columns.duplicated()].copy()

        rating_long = self.get_rating_long().merge(final_res.head(1), on='flid', how='inner', indicator=True)

        res = {
            "flid": rating_long.head(1)['flid'].iloc[0],
            "x": rating_long.head(1)['x'].iloc[0],
            "y": rating_long.head(1)['y'].iloc[0],
            "deck": rating_long.head(1)['deck'].iloc[0],
        }
        return res