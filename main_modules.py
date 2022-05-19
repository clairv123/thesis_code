import inspect 
import re
from time import time
import datetime
import hashlib
from google.cloud import firestore
import datetime



def map_to_dict(item):
    dictionary = dict(item)

    for k, v in dictionary.items():
        if type(v) == datetime.datetime:
            dictionary[k] = v.strftime("%Y-%m-%dT%H:%M:%S")
        # FireStore does not like Date Types.
        if type(v).__name__ == "date":
            dictionary[k] = v.strftime("%Y-%m-%dT00:00:00")
    return dictionary

# Declared at cold-start, but only initialized if/when the function executes
time_new = time()  # log execution time

class logger:
    def __init__(self,message):
        global time_new
        self.tstamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.message = re.sub('<[^<]+?> |<[^<]+?>', '', message)  # strip HTML tags
        time_old = time_new
        time_new = time()
        sec = time_new - time_old  # log execution time
        print(f'{self.tstamp} | {sec: >5.1f}s | {inspect.stack()[1][3]} | {self.message}')


def divide_chunks(l: list, n: int = 500) -> list:
    """""
    Divides the list in chunks of N.
    """""
    logger("Chunking.")
    for i in range(0, len(l), n): 
        yield l[i:i + n]
    logger("Done chunking.")


def force_to_dict(request):
    if type(request) is not dict:
        request = request.get_json(force=True)
    return request 

def hash_string(string: str) -> str:
    return hashlib.sha256(string.lower().encode('utf-8')).hexdigest() if type(string) is str else None

def scrape_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")


def clean_null_terms(d: dict) -> dict:
    """
    Cleans out all key-value pairs with a Null/None value to prevent sending superfluous data.
    """
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = clean_null_terms(v)
            if len(nested.keys()) > 0:
                clean[k] = nested
        elif v is not None:
            clean[k] = v
    return clean


def define_key_value_pairs(object):
    for k, v in object.items(): 
        if isinstance(v, dict):
            define_key_value_pairs(v)
            object[k] = [{"key": k, "value": v} for k, v in object[k].items()]
    return object

def stream_array_to_firestore(collection_name: str, document_name: str, event: str) -> str:
        DB = firestore.Client()
        
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






def stream_to_firestore(collection_name: str, document_name: str, data: dict) -> str:
        DB = firestore.Client()

        data["updated"] = datetime.datetime.now(tz=datetime.timezone.utc)

        # FireStore does not like Date Types.
        for k, v in data.items():
            if type(v).__name__ == "date":
                data[k] = v.strftime("%Y-%m-%dT00:00:00")
        
        DB.collection(collection_name).document(document_name).set(data)

        logger("Streamed to FireStore.")

        return "Streamed"