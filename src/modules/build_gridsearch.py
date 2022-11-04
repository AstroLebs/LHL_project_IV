from sklearn.model_selection import GridSearchCV


class build:
    def __init__(self, pipeline, params=None):
        self.pipe = pipeline
        linreg_params = {
            "reduce_dimensionality__n_components": [5, 10, None],
            # "classifier": [self.pipe.get_model("linear")],
        }
        logreg_params = {
            "reduce_dimensionality__n_components": [5, 10, None],
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.1, 1, 10],
            # "classifier": [self.pipe.get_model("log")],
        }
        rf_params = {
            "reduce_dimensionality__n_components": [5, 10, None],
            "classifier__n_estimators": [100, 200],
            "classifier__min_samples_leaf": [1, 2],
            "classifier": [self.pipe.get_classifier("r_forest")],
        }
        self.default_params = [rf_params]
        self.params = params if params is not None else self.default_params

    def search(self, pipe):
        return GridSearchCV(self.pipe, self.params)
