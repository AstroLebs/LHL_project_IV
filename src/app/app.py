import pickle

import pandas as pd
from flask import Flask, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

model = pickle.load(open("../data/pickles/model.p", "rb"))


@app.route("/predict")
def post(self):
    json_data = request.get_json()
    df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
    res = model.predict(df)
    return res.to_list()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
