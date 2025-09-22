from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from benchmark import benchmark_dataset


if __name__ == "__main__":
    iris = load_iris()
    benchmark_dataset(
        "Iris",
        iris.data,
        iris.target,
        ham_params={"n_tables": 6, "n_bits": 8, "random_state": 42},
        ann_params={
            "hidden_layer_sizes": (32, 16),
            "max_iter": 500,
            "random_state": 42,
        },
    )

    wine = load_wine()
    benchmark_dataset(
        "Wine",
        wine.data,
        wine.target,
        ham_params={"n_tables": 10, "n_bits": 10, "random_state": 42},
        ann_params={
            "hidden_layer_sizes": (64, 32),
            "max_iter": 500,
            "random_state": 42,
        },
    )

    bc = load_breast_cancer()
    benchmark_dataset(
        "Breast Cancer",
        bc.data,
        bc.target,
        ham_params={"n_tables": 12, "n_bits": 12, "random_state": 42},
        ann_params={"hidden_layer_sizes": (128,), "max_iter": 500, "random_state": 42},
    )
