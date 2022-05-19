# Before running do the following:

## gcloud init 
## Create a new initialization for your project


# --source= . \

gcloud functions deploy probability_calculator \
--region=europe-west1 \
--allow-unauthenticated \
--entry-point=main \
--memory=2048MB \
--runtime=python38 \
--service-account=
--trigger-http \
--timeout=180


gcloud functions deploy probability_calculator_fs_trigger \
--runtime=python38 \
--service-account=sandbox-clairv@appspot.gserviceaccount.com \
--region=europe-west1 \
--entry-point=main \
--memory=2048MB \
--timeout=180 \
--trigger-event=providers/cloud.firestore/eventTypes/document.write \
--trigger-resource="


