%%bash
export OUTDIR='gs://cloud_upload_training_ml01/flights/trained_model'
echo $OUTDIR

model_dir=$(gsutil ls ${OUTDIR}/export | tail -1)
echo $model_dir
saved_model_cli show --tag_set serve --signature_def serving_default --dir $model_dir