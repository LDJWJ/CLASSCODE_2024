%%bash

export OUTDIR='gs://cloud_upload_training_ml01/flights/trained_model'
echo $OUTDIR
gsutil -m rm -rf $OUTDIR

# export OUTDIR='gs://[버킷명]/flights/trained_model'
# 저장 후, chmod 755 c08.sh
# ./c08.sh