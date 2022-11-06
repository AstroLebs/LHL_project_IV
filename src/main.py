import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from modules import build_pipeline
import pickle
import constants


def main():
    """
    Runs pipeline and saves the best estimator as found by GridSearchCV as a pickle
    """
    df = pd.read_csv("data/data.csv")

    X = df.drop("Loan_Status", axis=1)
    y = df.Loan_Status
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=constants.random_state
    )

    pipe = build_pipeline.build(X_train, y_train, droplist=["Loan_ID"])
    pipe = pipe.complete_pipeline()

    print(cross_val_score(pipe, X_train, y_train, cv=10, scoring="accuracy").mean())

    grid = GridSearchCV(pipe, constants.params, cv=10, scoring="accuracy", verbose=3)
    grid.fit(X_train, y_train)

    print(grid.best_score_)
    print(grid.best_params_)

    model = grid.best_estimator_

    with open("pickles/model.p", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
