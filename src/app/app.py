import pickle

import pandas as pd
from flask import Flask, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

model = pickle.load(open("../data/pickles/model.p", "rb"))


class RawFeats:
    # Move to Functions File
    def __init__(self, feats):
        self.feats = feats

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return X[self.feats]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        res = model.predict_proba(df)
        return res.to_list()


# assign endpoints
api.add_resource(Predict, "/Predict")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
