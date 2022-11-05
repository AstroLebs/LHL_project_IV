"""
constants.py
"""


n_jobs = -1
random_state = 3791


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

clf1 = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
clf2 = LogisticRegression(n_jobs=n_jobs, random_state=random_state)
clf3 = KNeighborsClassifier(n_jobs=n_jobs)


params1 = {
    "reduce_dimensionality__n_components": [5, 10, 15, 20, None],
    "classifier__n_estimators": [100, 150, 200, 250, 300],
    "classifier__min_samples_leaf": [1, 2, 3],
    "classifier": [clf1],
}

params2 = {
    "reduce_dimensionality__n_components": [5, 10, 15, 20, None],
    "classifier": [clf2],
}

params3 = {
    "reduce_dimensionality__n_components": [5, 10, 15, 20, None],
    "classifier__n_neighbors": [3, 5, 10, 15],
    "classifier__weights": ["uniform", "distance"],
    "classifier": [clf3],
}

params = [params1, params2, params3]
