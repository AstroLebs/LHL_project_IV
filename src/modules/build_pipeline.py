from sklearn.decomposition import FactorAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
import constants


class build:
    def __init__(self, data, target, droplist=None):
        self.data = data if droplist is None else data.drop(droplist, axis=1)
        self.target = target.str.strip().eq("Y").mul(1)
        self.categorical_features = data.select_dtypes(
            exclude="number"
        ).columns.tolist()
        self.quantitative_features = data.drop(
            self.categorical_features, axis=1
        ).columns.tolist()
        self.pipe = None

    def build_preprocessor(self):
        cat_trans = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ("imputer", KNNImputer()),
            ]
        )

        quant_trans = Pipeline(
            steps=[
                ("imputer", KNNImputer()),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("quantitative", quant_trans, self.quantitative_features),
                ("categorical", cat_trans, self.categorical_features),
            ],
            sparse_threshold=0,
            remainder="drop",
        )

        return preprocessor

    def get_classifier(self, key):
        return self.models[key]

    def complete_pipeline(self):
        clf1 = RandomForestClassifier(random_state=constants.random_state)
        pipe = Pipeline(
            [
                ("preprocessor", self.build_preprocessor()),
                (
                    "reduce_dimensionality",
                    FactorAnalysis(random_state=constants.random_state),
                ),
                ("classifier", clf1),
            ]
        )
        self.pipe = pipe
        return pipe

    def fit(self, X, y):
        y = y.str.strip().eq("Y").mul(1)
        self.pipe.fit(X, y)
        return self.pipe

    def predict(self, X):
        return self.pipe.predict(X)

    def fit_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def get_self(self):
        return self
