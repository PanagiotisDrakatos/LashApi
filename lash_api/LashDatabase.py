import tensorflow as tf
import pandas as pd

class DatabaseExtract:
    def __init__(self, oids):
        self.oids = oids
        self.__FINGERPRINT_LOCATION_PATH = "C:\\Users\\User\\PycharmProjects\\lashProject\\FINGERPRINT_LOCATION.csv"
        self.__FINGERPRINT_LOCALIZE_TEMP_PATH = "C:\\Users\\User\\PycharmProjects\\lashProject\\FINGERPRINT_LOCALIZE_TEMP.csv"
        self.__FINGERPRINT_OBJECT_PATH = "C:\\Users\\User\\PycharmProjects\\lashProject\\FINGERPRINT_OBJECT.csv"

    def __init__(self):
        self.__FINGERPRINT_LOCATION_PATH = "C:\\Users\\User\\PycharmProjects\\lashProject\\FINGERPRINT_LOCATION.csv"
        self.__FINGERPRINT_LOCALIZE_TEMP_PATH = "C:\\Users\\User\\PycharmProjects\\lashProject\\FINGERPRINT_LOCALIZE_TEMP.csv"
        self.__FINGERPRINT_OBJECT_PATH = "C:\\Users\\User\\PycharmProjects\\lashProject\\FINGERPRINT_OBJECT.csv"

    def get_oid(self):
        return self.oids


    def set_oid(self, oids):
        self.oids = oids

    def get_panda_dataset(self,file_path):
        dataset = pd.read_csv(file_path, index_col=None)
        del dataset['uid']
        del dataset['timestr']
        del dataset['buid']
        return dataset

    def __get_dataset(self,file_path, LABEL_COLUMN="uid", col_types=None):
        df = pd.read_csv(file_path, index_col=None, encoding='latin-1')
        dataset = tf.data.experimental.make_csv_dataset(
            file_path,
            batch_size=df.size,  # Artificially small to make examples easier to show.
            column_defaults=col_types,
            label_name=LABEL_COLUMN,
            select_columns=None,
            field_delim=',',
            use_quote_delim=True,
            na_value="?",
            header=True,
            num_epochs=1,
            ignore_errors=True,
            shuffle=True,
            shuffle_buffer_size=10000,
            shuffle_seed=None,
            prefetch_buffer_size=None,
            num_parallel_reads=2,
            sloppy=False,
            num_rows_for_inference=100,
            compression_type=None,
            encoding='latin-1')
        return dataset

    def __get_uniques(self,t):
        t1d = tf.reshape(t, shape=(-1,))
        # or tf.unique, if you don't need counts
        uniques, idx = tf.unique(t1d)
        return uniques

    def __process_categorical_data(self,data):
        """Returns a one-hot encoded tensor representing categorical values."""
        regex = r"(<br/>)|(<EOF>)|(<SOF>)|[\n\!\@\#\$\%\^\&\*\(\)\[\]\
                   {\}\;\:\,\.\/\?\|\`\_\\+\\\=\~\-\<\>]"
        regex2 = r'\r\n(\r\n|\s\s\s)'
        # Remove leading ' '.
        data = tf.strings.regex_replace(data, '^ ', '')
        # Remove trailing '.'.
        data = tf.strings.regex_replace(data, r'\.$', '')

        data = tf.strings.regex_replace(data, 'b', '')
        data = tf.strings.regex_replace(data, regex, '')
        data = tf.strings.regex_replace(data, regex2, '')

        data = tf.strings.unicode_decode(data, 'UTF-16-BE', errors='ignore')
        data = tf.strings.unicode_encode(data, 'UTF-16-BE', errors='ignore')

    def ClosestValue(self,t, closest_neighbors):
        counter = 0
        for c in closest_neighbors:
            tensor = tf.math.squared_difference(t, c)
            indices = tf.math.argmin(tensor, axis=0)
            a = tensor[indices[0], 0]
            b = tensor[indices[1], 1]
            final_indices = tf.where(tf.less(a, b), [indices[0], 0], [indices[1], 1])
            closest_value = tf.gather_nd(t, final_indices)
            if counter == 0:
                x_cord = closest_value
                print('Closest X value to {} is {}'.format(c, x_cord))
            elif counter == 1:
                y_cord = closest_value
                print('Closest Y value to {} is {}'.format(c, y_cord))
            counter = counter + 1
        return [x_cord, y_cord]


    def extraction_flid(self):
        FINGERPRINT_LOCATION_LABEL_COLUMN = 'uid'
        FINGERPRINT_LOCALIZE_TEMP_LABEL_COLUMN = 'uid'
        FINGERPRINT_OBJECT_LABEL_COLUMN = 'foid'
        FINGERPRINT_LOCATION_TYPES = [tf.int64, tf.string, tf.int64, tf.string, tf.string, tf.float64, tf.float64,
                                      tf.int64,
                                      tf.int64]
        FINGERPRINT_OBJECT_TYPES = [tf.int32, tf.int32, tf.int32, tf.float64, tf.float64, tf.string]

        FINGERPRINT_LOCATION = self.__get_dataset(self.__FINGERPRINT_LOCATION_PATH, FINGERPRINT_LOCATION_LABEL_COLUMN,
                                           FINGERPRINT_LOCATION_TYPES)
        FINGERPRINT_OBJECT = self.__get_dataset(self.__FINGERPRINT_OBJECT_PATH, FINGERPRINT_OBJECT_LABEL_COLUMN)
        FINGERPRINT_LOCALIZE_TEMP = self.__get_dataset(self.__FINGERPRINT_LOCALIZE_TEMP_PATH, FINGERPRINT_LOCALIZE_TEMP_LABEL_COLUMN)
        FINGERPRINT_LOCALIZE_TEMP.apply(tf.data.experimental.ignore_errors())

        FINGERPRINT_LOCATION_MODELS, FINGERPRINT_LOCATION_LABELS = next(
            iter(FINGERPRINT_LOCATION))  # Just the first batch. # Just the first batch.
        FINGERPRINT_LOCALIZE_TEMP_MODELS, FINGERPRINT_LOCALIZE_TEMP_LABELS = next(iter(FINGERPRINT_LOCALIZE_TEMP))
        FINGERPRINT_OBJECT_MODELS, FINGERPRINT_OBJECT_LABELS = next(iter(FINGERPRINT_OBJECT))

        # dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3, 4],
        #                                              'b': [5, 6,7,8]})

        vals = []
        vals3 = []
        for idx, x in enumerate(self.oids):
            vals.append([idx])
            vals3.append(self.oids[idx])

        indices = tf.constant(vals)
        updates = tf.constant(vals3)

        ###############FINGERPRINT_LOCALIZE_TEMP table#########################
        OID = tf.constant(tf.cast(FINGERPRINT_LOCALIZE_TEMP_MODELS['oid'], tf.float32))
        HEIGHT = tf.constant(FINGERPRINT_LOCALIZE_TEMP_MODELS['height'])
        WIDTH = tf.constant(FINGERPRINT_LOCALIZE_TEMP_MODELS['width'])
        OCR = tf.constant(FINGERPRINT_LOCALIZE_TEMP_MODELS['ocr'])

        OID = tf.reshape(OID, [tf.size(OID), 1])
        HEIGHT = tf.reshape(HEIGHT, [tf.size(HEIGHT), 1])
        WIDTH = tf.reshape(WIDTH, [tf.size(WIDTH), 1])
        OCR = tf.reshape(OID, [tf.size(OCR), 1])

        FINGERPRINT_LOCALIZE_MERGE = tf.concat(axis=1, values=[OID, HEIGHT, WIDTH, OCR])

        shape = tf.constant([FINGERPRINT_LOCALIZE_MERGE.get_shape().dims[0]])
        scatter = tf.scatter_nd(indices, updates, shape)

        FINGERPRINT_LOCALIZE_INDICES = tf.where(tf.equal(tf.cast(OID, tf.int32), scatter));
        FINGERPRINT_LOCALIZE_RESULT = tf.unstack(FINGERPRINT_LOCALIZE_INDICES, axis=1)
        FINGERPRINT_LOCALIZE_MERGE2 = tf.gather(FINGERPRINT_LOCALIZE_MERGE, FINGERPRINT_LOCALIZE_RESULT)

        ###############FINGERPRINT_LOCALIZE_TEMP table#########################

        ####################FINGERPRINT_OBJECT_MODELS#################
        FLID = tf.constant(tf.cast(FINGERPRINT_OBJECT_MODELS['flid'], tf.float64))
        OID2 = tf.constant(tf.cast(FINGERPRINT_OBJECT_MODELS['oid'], tf.float64))
        HEIGHT2 = tf.constant(tf.cast(FINGERPRINT_OBJECT_MODELS['height'], tf.float64))
        WIDTH2 = tf.constant(tf.cast(FINGERPRINT_OBJECT_MODELS['width'], tf.float64))
        OCR2 = tf.constant(FINGERPRINT_OBJECT_MODELS['ocr'])

        FLID = tf.reshape(FLID, [tf.size(FLID), 1])
        OID2 = tf.reshape(OID2, [tf.size(OID2), 1])
        HEIGHT2 = tf.reshape(HEIGHT2, [tf.size(HEIGHT2), 1])
        WIDTH2 = tf.reshape(WIDTH2, [tf.size(WIDTH2), 1])
        OCR2 = tf.reshape(OCR2, [tf.size(OCR2), 1])

        FINGERPRINT_OBJECT_CONCAT = tf.concat(axis=1, values=[FLID, OID2, HEIGHT2, WIDTH2])
        FINGERPRINT_OBJECT_CONCAT2 = tf.concat(axis=1, values=[FLID])

        FINGERPRINT_OBJECT_INDICES = tf.where(tf.equal(tf.cast(OID2, tf.int32), scatter));
        FINGERPRINT_OBJECT_RESULT = tf.unstack(FINGERPRINT_OBJECT_INDICES, axis=1)
        FINGERPRINT_OBJECT_MERGE = tf.gather(FINGERPRINT_OBJECT_CONCAT, FINGERPRINT_OBJECT_RESULT)
        FINGERPRINT_OBJECT_MERGE2 = tf.gather(FINGERPRINT_OBJECT_CONCAT2, FINGERPRINT_OBJECT_RESULT)

        # MATCED_FLID = tf.transpose(tf.nn.embedding_lookup(tf.transpose(FINGERPRINT_OBJECT_MERGE2), tf.constant([0])))
        MATCED_FLID = tf.split(FINGERPRINT_OBJECT_MERGE2, num_or_size_splits=FINGERPRINT_OBJECT_MERGE.shape.dims[1],
                               axis=1)
        MATCED_FLID = tf.concat([MATCED_FLID], 1)
        MATCED_FLID = tf.unstack(MATCED_FLID, axis=1)
        del MATCED_FLID[1]
        MATCED_FLID = tf.stack(MATCED_FLID, 1)
        MATCED_FLID = self.__get_uniques(MATCED_FLID)
        # val=FINGERPRINT_OBJECT_MERGE.shape.dims[1]
        # MATCED_FLID=tf.unstack(FINGERPRINT_OBJECT_MERGE, axis=1)
        # MATCED_FLID = tf.unstack(MATCED_FLID, axis=1)
        # MATCED_FLID = tf.unstack(MATCED_FLID, axis=1)
        # MATCED_FLID=tf.reshape(FINGERPRINT_OBJECT_MERGE, [-1])
        ####################FINGERPRINT_OBJECT table#################

        #########FINGERPRINT_LOCATION###################
        FLID3 = tf.constant(tf.cast(FINGERPRINT_LOCATION_MODELS['flid'], tf.float64))
        X = tf.constant(tf.cast(FINGERPRINT_LOCATION_MODELS['x'], tf.float64))
        Y = tf.constant(tf.cast(FINGERPRINT_LOCATION_MODELS['y'], tf.float64))
        DECK = tf.constant(tf.cast(FINGERPRINT_LOCATION_MODELS['deck'], tf.float64))

        FLID3 = tf.reshape(FLID3, [tf.size(FLID3), 1])
        X = tf.reshape(X, [tf.size(X), 1])
        Y = tf.reshape(Y, [tf.size(Y), 1])
        DECK = tf.reshape(DECK, [tf.size(DECK), 1])

        FINGERPRINT_LOCATION_MERGE = tf.concat(axis=1, values=[FLID3, X, Y, DECK])
        FINGERPRINT_LOCATION_MIRROR = tf.concat(axis=1, values=[X, Y])

        # df=tf.reshape(MATCED_FLID,shape=(510,))
        FINGERPRINT_LOCATION_INDICES = tf.where(tf.equal(tf.cast(FLID3, tf.double), MATCED_FLID));
        FINGERPRINT_LOCATION_RESULT = tf.unstack(tf.cast(FINGERPRINT_LOCATION_INDICES, tf.int32), axis=1)
        del FINGERPRINT_LOCATION_RESULT[1]
        FINGERPRINT_LOCATION_MERGE2 = tf.gather(FINGERPRINT_LOCATION_MERGE, self.__get_uniques(FINGERPRINT_LOCATION_RESULT))

        splitted=tf.unstack(FINGERPRINT_LOCATION_MERGE2, axis=1)
        flid_array=splitted[0].numpy().tolist()
        x_array=splitted[1].numpy().tolist()
        y_array = splitted[2].numpy().tolist()
        z_array = splitted[3].numpy().tolist()

        return flid_array,x_array,y_array,z_array