import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from classifier import Hash_based_Classifier


def benchmark_dataset(name, X, y, ham_params={}, ann_params={}):
    print(f"\n=== {name} 데이터셋 ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Hash_based_Classifier
    ham = Hash_based_Classifier(**ham_params)
    t0 = time.time()
    ham.fit(X_train_s, y_train)
    t1 = time.time()
    yhat_h = ham.predict(X_test_s)
    t2 = time.time()

    print("Hash_based_Classifier")
    print("  Accuracy :", accuracy_score(y_test, yhat_h))
    print("  F1-macro :", f1_score(y_test, yhat_h, average="macro"))
    print("  Fit time :", round(t1 - t0, 6), "sec")
    print("  Pred time:", round(t2 - t1, 6), "sec")

    # ANN (MLP)
    ann = MLPClassifier(**ann_params)
    t0 = time.time()
    ann.fit(X_train_s, y_train)
    t1 = time.time()
    yhat_a = ann.predict(X_test_s)
    t2 = time.time()

    print("ANN (MLP)")
    print("  Accuracy :", accuracy_score(y_test, yhat_a))
    print("  F1-macro :", f1_score(y_test, yhat_a, average="macro"))
    print("  Fit time :", round(t1 - t0, 6), "sec")
    print("  Pred time:", round(t2 - t1, 6), "sec")
