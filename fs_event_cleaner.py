example_object = {'oldValue': {'createTime': '2022-05-11T20:15:31.475961Z', 'fields': {'events': {'arrayValue': {'values': [{'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-589518427'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:15:31.348086Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:15:26.525Z'}}}}, {'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-341756686'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:16:03.725644Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:15:54.215Z'}}}}, {'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-446543951'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:16:40.248289Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:16:29.023Z'}}}}, {'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-206632776'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:17:36.758656Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:17:34.389Z'}}}}]}}}, 'name': 'projects/jodp-stream/databases/(default)/documents/rtpc-webevents/193193888', 'updateTime': '2022-05-11T20:17:36.888340Z'}, 'updateMask': {'fieldPaths': ['events']}, 'value': {'createTime': '2022-05-11T20:15:31.475961Z', 'fields': {'events': {'arrayValue': {'values': [{'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-589518427'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:15:31.348086Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:15:26.525Z'}}}}, {'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-341756686'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:16:03.725644Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:15:54.215Z'}}}}, {'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-446543951'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:16:40.248289Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:16:29.023Z'}}}}, {'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-206632776'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:17:36.758656Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:17:34.389Z'}}}}, {'mapValue': {'fields': {'event': {'stringValue': 'add_to_cart'}, 'eventid': {'stringValue': '193193888-559836890'}, 'firestore_event_timestamp': {'timestampValue': '2022-05-11T20:18:14.115217Z'}, 'jodpsid': {'stringValue': '193193888'}, 'jodpuid': {'stringValue': 'J572336737-1652038096'}, 'web_event_timestamp': {'stringValue': '2022-05-11T20:18:02.710Z'}}}}]}}}, 'name': 'projects/jodp-stream/databases/(default)/documents/rtpc-webevents/193193888', 'updateTime': '2022-05-11T20:18:14.172920Z'}}
 


events = example_object.get("value", {}).get("fields", {}).get("events", {}).get("arrayValue", {}).get("values")

events_list = []
for event in events:
    mapvalue = event.get("mapValue")
    fields = mapvalue.get("fields")
    
    event_dict = {}
    for k, v in fields.items():
        for v in v.values():
            event_dict[k] = v
    events_list.append(event_dict)

            

# events = [event.get("event") for event in events_list]

# print(events)
