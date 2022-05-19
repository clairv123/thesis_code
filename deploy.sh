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
--service-account=sandbox-clairv@appspot.gserviceaccount.com \
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
--trigger-resource="projects/sandbox-clairv/databases/(default)/documents/webevent/{sessionid}"


# SANDBOX CLAIR
gcloud builds submit --tag eu.gcr.io/sandbox-clairv/probability-calculator


gcloud run deploy  probability-calculator  --image eu.gcr.io/sandbox-clairv/probability-calculator  --platform=managed\
    --service-account=510185108287-compute@developer.gserviceaccount.com\
    --update-labels=creator=davy_van_der_horst,application=probability-calculator\
    --memory=8Gi --timeout=3600 --region=europe-west1 --cpu=4


# JODP STREAM
gcloud builds submit --tag eu.gcr.io/jodp-stream/probability-calculator


gcloud run deploy  probability-calculator  --image eu.gcr.io/jodp-stream/probability-calculator  --platform=managed\
    --service-account=jodp-stream@appspot.gserviceaccount.com\
    --update-labels=creator=davy_van_der_horst,application=probability-calculator\
    --memory=8Gi --timeout=3600 --region=europe-west1 --cpu=4