from google.cloud import firestore
import datetime
from os import environ

# environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/davyvanderhorst/Desktop/Code/VisualCode/Macbook_2020/Sandbox/Clair/Probability Calculator/probability-calculator/Credentials/FireStore Service Account.json"
DB = firestore.Client()

def read_from_firestore(document_name): 
    doc_ref = DB.collection('webevents').document(document_name)

    doc = doc_ref.get()

    if doc.exists:
        document = doc.to_dict()
        events = document.get("events")

    return locals().get("events")


# print(locals())

# event_array = [item.get("event") for item in read_from_firestore("clair")]

# print(event_array)
