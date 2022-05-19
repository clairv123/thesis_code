from google.cloud import firestore
import datetime

DB = firestore.Client()

def read_from_firestore(document_name): 
    doc_ref = DB.collection('webevents').document(document_name)

    doc = doc_ref.get()

    if doc.exists:
        document = doc.to_dict()
        events = document.get("events")

    return locals().get("events")
