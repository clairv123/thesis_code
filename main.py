from os import path, environ
import numpy as np
from clean_csv import get_training_set
import joblib
import time
import gcsfs
from main_modules import scrape_time, stream_to_firestore
from stream_to_bq import stream_to_bq

# from os import environ
# environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/davyvanderhorst/Desktop/Code/VisualCode/Macbook_2020/Jumbo/JODP Stream/Service Accounts/App Engine default service account.json"


root = path.dirname(path.abspath(__file__))
bq_endpoint = "https://europe-west1-ux-datasources.cloudfunctions.net/stream_to_bigquery"

def convert_firestore_event_context(context):
    events = context.get("value", {}).get("fields", {}).get("events", {}).get("arrayValue", {}).get("values")
    event_list = []
    for event in events:
        mapvalue = event.get("mapValue", {})
        fields = mapvalue.get("fields", {})
        event = fields.get("event", {})
        value = event.get("stringValue")
        event_list.append(value)

    return event_list


def parse_events_list(request_data):
    return [item.get("event") for item in request_data.get("events")]


def get_file(file_name: str):      
    fs = gcsfs.GCSFileSystem(project="jodp-stream")
    
    bucket_name = "jumbo_rtpc_trained_models"
    bucket_path = f"{bucket_name}/{file_name}"
    
    file =  fs.open(bucket_path, 'rb')
    return file


def joblib_load(file_name: str):
    return joblib.load(file_name)

def lstm_model(data: dict) -> None:
    npy_file = get_file("lstm_model.npy")
    events = parse_events_list(data)
    
    LSTM_model = joblib_load(npy_file)
    seq_len = LSTM_model.layers[0].output_shape[1]

    X_test = get_training_set(events, seq_len)
    print(X_test.shape)
    print(X_test)

    samplerow = X_test
    sample = np.array([np.pad(samplerow,((i,0)), mode='constant')[:-i if i>0 else samplerow.shape[0]] for i in range(seq_len)])[::-1]

    print("LSTM:")
    LSTM_starttime = time.perf_counter()
    LSTM_SampleOutput = LSTM_model.predict(sample)
    LSTM_endtime = time.perf_counter()
    lstm_output = LSTM_SampleOutput[-np.trim_zeros(samplerow).shape[0]:]
    print(lstm_output)
    lstm_latest_probability = lstm_output[0][0].item() # Item converts the numpy.float to a native Python type.
    print(f'LSTM calculation time:{LSTM_endtime - LSTM_starttime:0.4f} seconds')

    event_data = data.get("events")
    final_event = event_data[-1]
    session_id = final_event.get("jodpsid")
    firestore_data = {"lstm_latest_probability": lstm_latest_probability}
    
    stream_to_firestore("rtpc-lstm-model", session_id, firestore_data)
    bigquery_payload = final_event | firestore_data
    bigquery_payload.pop("updated")
    bigquery_payload["processed_time"] = scrape_time()
    stream_to_bq(bigquery_payload, bq_endpoint, "realtime_conversion_probability", "lstm_model")
    


def rnn_model(data: dict) -> None:
    npy_file = get_file("rnn_model.npy")
    events = parse_events_list(data)
    
    RNN_model = joblib_load(npy_file)
    seq_len = RNN_model.layers[0].output_shape[1]

    X_test = get_training_set(events, seq_len)
    print(X_test.shape)
    print(X_test)

    samplerow = X_test
    sample = np.array([np.pad(samplerow,((i,0)), mode='constant')[:-i if i>0 else samplerow.shape[0]] for i in range(seq_len)])[::-1]

    print("RNN:")
    RNN_starttime = time.perf_counter()
    RNN_SampleOutput = RNN_model.predict(sample)
    RNN_endtime = time.perf_counter()
    rnn_output = RNN_SampleOutput[-np.trim_zeros(samplerow).shape[0]:]
    print(rnn_output)
    rnn_latest_probability = rnn_output[0][0].item()
    print(f'RNN calculation time:{RNN_endtime - RNN_starttime:0.4f} seconds')

    event_data = data.get("events")
    final_event = event_data[-1]
    session_id = final_event.get("jodpsid")
    firestore_data = {"rnn_latest_probability": rnn_latest_probability}
    
    stream_to_firestore("rtpc-rnn-model", session_id, firestore_data)
    bigquery_payload = final_event | firestore_data
    bigquery_payload.pop("updated")
    bigquery_payload["processed_time"] = scrape_time()
    stream_to_bq(bigquery_payload, bq_endpoint, "realtime_conversion_probability", "rnn_model")




def gru_model(data: dict) -> None:
    full_file_path_lstm = get_file("gru_model.npy")
    events = parse_events_list(data)

    GRU_model = joblib_load(full_file_path_lstm)

    seq_len = GRU_model.layers[0].output_shape[1]

    X_test = get_training_set(events, seq_len)
    print(X_test.shape)
    print(X_test)

    samplerow = X_test
    sample = np.array([np.pad(samplerow,((i,0)), mode='constant')[:-i if i>0 else samplerow.shape[0]] for i in range(seq_len)])[::-1]

    print("GRU:")
    GRU_starttime = time.perf_counter()
    GRU_SampleOutput = GRU_model.predict(sample)
    GRU_endtime = time.perf_counter()
    gru_output = GRU_SampleOutput[-np.trim_zeros(samplerow).shape[0]:]
    print(gru_output)
    gru_latest_probability = gru_output[0][0].item()
    print(f'GRU calculation time:{GRU_endtime - GRU_starttime:0.4f} seconds')

    event_data = data.get("events")
    final_event = event_data[-1]
    session_id = final_event.get("jodpsid")
    firestore_data = {"gru_latest_probability": gru_latest_probability}
    
    stream_to_firestore("rtpc-gru-model", session_id, firestore_data)
    bigquery_payload = final_event | firestore_data
    bigquery_payload.pop("updated")
    bigquery_payload["processed_time"] = scrape_time()
    stream_to_bq(bigquery_payload, bq_endpoint, "realtime_conversion_probability", "gru_model")
    




def main(events_list: list, model):
    data = events_list

    runned_model = model(data)

    return "OK"
