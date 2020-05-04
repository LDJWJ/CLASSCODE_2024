## ===========
## Versino Check
## ===========
import tensorflow as tf
import sys
print(tf.version.VERSION)
print('{}.{}'.format(sys.version_info.major,sys.version_info.minor))

## ===========
## SETUP
## ===========
# change these to try this notebook out
# In "production", these will be replaced by the parameters passed to papermill
BUCKET = "cloud_upload_training_ml01"
PROJECT = "apt-diode-156508"
REGION = "us-east1"
DEVELOP_MODE = True
NBUCKETS = 5 # for embeddings
NUM_EXAMPLES = 1000*1000 # assume 1 million examples
TRAIN_BATCH_SIZE = 64
DNN_HIDDEN_UNITS = '64,32'

import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION

# upload file
DATA_BUCKET = "gs://{}/flights/chapter8/output/".format(BUCKET)
TRAIN_DATA_PATTERN = DATA_BUCKET + "train01.csv"
EVAL_DATA_PATTERN = DATA_BUCKET + "test01.csv"

# python03 c03.py