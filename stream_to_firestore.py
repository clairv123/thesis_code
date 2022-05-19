from google.cloud import firestore
import datetime

from os import environ
environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/davyvanderhorst/Desktop/Code/VisualCode/Macbook_2020/Sandbox/Clair/Probability Calculator/probability-calculator/Credentials/FireStore Service Account.json"

DB = firestore.Client()

def stream_to_firestore(collection_name: str, document_name: str, event: str) -> str:
        
        events_key = "events"

        data = {
        "event_timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
        "event": event
        }
        doc_ref = DB.collection(collection_name).document(document_name)
        try:
            doc_ref.update({events_key: firestore.ArrayUnion([data])})
        except Exception as e:
            doc_ref.set({events_key: [data]})


        return "Streamed"


event_enum = ['homepage', 'list_view',  'item_search', 'product_click', 'add_to_basket', 'checkout_step']

for i in range(10):
    print(i)
    for event in event_enum:
        stream_to_firestore("webevents", "clair15", event)

# stream_to_firestore("webevents", "clair5", event_enum[0])