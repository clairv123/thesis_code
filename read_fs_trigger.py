import json
import requests
import threading

def force_to_dict(request):
    if type(request) is not dict:
        request = request.get_json(force=True)
    return request 


def send_array(events, endpoint= "/"):
    url = 

    payload = json.dumps({
    "events": events
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    
    return "OK"


def convert_firestore_event_context(event, context):

    event_dict = force_to_dict(event)

    events = event_dict.get("value", {}).get("fields", {}).get("events", {}).get("arrayValue", {}).get("values")

    event_list = []
    for event in events:
        mapvalue = event.get("mapValue", {})
        fields = mapvalue.get("fields", {})
        event = fields.get("event", {})
        value = event.get("stringValue")
        event_list.append(value)

    print(event_list)

    return "OK"

    endpoints = ["/lstm_model", "/rnn_model", "/gru_model"]
    threads = []
    for endpoint in endpoints:
        t = threading.Thread(target=send_array, args=[event_list, endpoint])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    return "OK"
