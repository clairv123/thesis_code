from google.cloud import firestore
import datetime
import json
import base64
import flask


DB = firestore.Client()

def force_to_dict(request):
    if type(request) is not dict:
        request = request.get_json(force=True)
    return request 


def stream_to_firestore(event, context) -> str:


        request = json.loads(base64.b64decode(
                event['data']).decode('utf-8'))["jsonPayload"]

        request = force_to_dict(request)
        
        jodpsid = request.get("jodpsid") # Session ID
        jodpuid = request.get("jodpuid") # User ID
        document_name = jodpsid
        

        data = {
        "firestore_event_timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
        "web_event_timestamp": request.get("timestamp"),
        "event": request.get("event", "").lower(),
        "eventid": request.get("eventid"),
        "jodpsid": jodpsid,
        "jodpuid": jodpuid

        }
        doc_ref = DB.collection("rtpc-webevents").document(document_name)
        try:
            doc_ref.update({"events": firestore.ArrayUnion([data])})
        except Exception as e:
            doc_ref.set({"events": [data]})


        return flask.Response(response=json.dumps({"message": "OK"}), status=200)