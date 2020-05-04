### SETUP
BUCKET = "cloud_upload_training_ml01"
PROJECT = "apt-diode-156508"
REGION = "us-east1"

DEVELOP_MODE = True
NBUCKETS = 5 # for embeddings
NUM_EXAMPLES = 1000*1000 # assume 1 million examples
TRAIN_BATCH_SIZE = 64
DNN_HIDDEN_UNITS = '64,32'

DATA_BUCKET = "gs://{}/flights/chapter8/output/".format(BUCKET)
TRAIN_DATA_PATTERN = DATA_BUCKET + "train01.csv"
EVAL_DATA_PATTERN = DATA_BUCKET + "test01.csv"

### Use tf.data to read the CSV files
import os, json, math, shutil
import numpy as np
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

### column
CSV_COLUMNS  = ('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + \
                ',carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'
DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]

def load_dataset(pattern, batch_size=1):
  return tf.data.experimental.make_csv_dataset(pattern, batch_size, CSV_COLUMNS, DEFAULTS)
  

if DEVELOP_MODE:
	dataset = load_dataset(TRAIN_DATA_PATTERN)
	for n, data in enumerate(dataset):
		print(n, data)
		numpy_data = {k: v.numpy() for k, v in data.items()} # .numpy() works only in eager mode
		print(numpy_data)
		if n>3: break


def features_and_labels(features):
  label = features.pop('ontime') # this is what we will train for
  return features, label

def read_dataset(pattern, batch_size, mode=tf.estimator.ModeKeys.TRAIN, truncate=None):
  dataset = load_dataset(pattern, batch_size)
  dataset = dataset.map(features_and_labels)
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(batch_size*10)
    dataset = dataset.repeat()
  dataset = dataset.prefetch(1)
  if truncate is not None:
    dataset = dataset.take(truncate)
  return dataset


if DEVELOP_MODE:
    print("Checking input pipeline")
    one_item = read_dataset(TRAIN_DATA_PATTERN, batch_size=2, truncate=1)
    print(list(one_item)) # should print one batch of 2 items


# python3 c05A.py
