import json
from flask import Flask, request
from flask_cors import CORS
from main_modules import force_to_dict

### main program
from main import main, lstm_model, rnn_model, gru_model

app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

app.config.update (
    # DEBUG = True,
    PROPAGATE_EXCEPTIONS = True
)

options_headers = {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST',
          'Access-Control-Allow-Headers': 'Content-Type',
          'Access-Control-Max-Age': '3600'
      }




@app.route('/', methods=['POST'])
def call_main():
    if request.method == 'OPTIONS':
      return ('', 204, options_headers)

    # The object request exists here, and contains the actual http request.
    request_dict = json.loads(request.data)
    request_dict = force_to_dict(request_dict)

    response = main(request_dict, lstm_model)
    response = main(request_dict, rnn_model)
    response = main(request_dict, gru_model)

    if response == "OK":
      return (str(response), 200, {})
    else:
      return (str(response), 500, {})


@app.route('/lstm_model', methods=['POST'])
def call_lstm():
    if request.method == 'OPTIONS':
      return ('', 204, options_headers)

    # The object request exists here, and contains the actual http request.
    request_dict = json.loads(request.data)
    request_dict = force_to_dict(request_dict)

    response = main(request_dict, lstm_model)

    if response == "OK":
      return (str(response), 200, {})
    else:
      return (str(response), 500, {})


@app.route('/rnn_model', methods=['POST'])
def call_rnn():
    if request.method == 'OPTIONS':
      return ('', 204, options_headers)

    # The object request exists here, and contains the actual http request.
    request_dict = json.loads(request.data)
    request_dict = force_to_dict(request_dict)

    response = main(request_dict, rnn_model)

    if response == "OK":
      return (str(response), 200, {})
    else:
      return (str(response), 500, {})


@app.route('/gru_model', methods=['POST'])
def call_gru():
    if request.method == 'OPTIONS':
      return ('', 204, options_headers)

    # The object request exists here, and contains the actual http request.
    request_dict = json.loads(request.data)
    request_dict = force_to_dict(request_dict)

    response = main(request_dict, gru_model)

    if response == "OK":
      return (str(response), 200, {})
    else:
      return (str(response), 500, {})


if __name__ == '__main__':
   app.run()
