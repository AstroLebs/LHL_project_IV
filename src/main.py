from modules import build_gridsearch, build_pipeline


def main():
    pipe = build_pipeline()
    grid = build_gridsearch()
    grid.fit(X, y)
    print(grid.best_params_)


if __name__ == "__main__":
    main()
