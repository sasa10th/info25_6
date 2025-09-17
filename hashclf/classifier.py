# -*- coding: utf-8 -*-
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.utils import check_random_state


class Hash_based_Classifier:
    """
    해시 기반 분류기 (Hash-based Classifier, HAM-Net).
    랜덤 프로젝션(Random Projection)과 해싱(Hashing)을 통해
    클래스별 연상 메모리 구조를 구축하는 단순·경량 분류기.

    Parameters
    ----------
    n_tables : int
        해시 테이블 개수.
    n_bits : int
        각 해시 테이블의 비트 수.
    learning_rate : float
        가중치 업데이트 학습률.
    decay : float
        가중치 감소율 (정규화).
    use_unary : bool
        단일 해시 비트(unary feature) 사용 여부.
    unary_weight : float
        단일 해시 비트 가중치.
    random_state : int
        난수 시드.
    """

    def __init__(
        self,
        n_tables=8,
        n_bits=10,
        learning_rate=1.0,
        decay=0.0,
        use_unary=True,
        unary_weight=0.2,
        random_state=42,
    ):
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.learning_rate = learning_rate
        self.decay = decay
        self.use_unary = use_unary
        self.unary_weight = unary_weight
        self.random_state = random_state

        self.proj_ = None
        self.classes_ = None
        self.W_ = {}
        self.B_ = {}

    def _build_proj(self, d):
        rng = check_random_state(self.random_state)
        self.proj_ = rng.normal(size=(self.n_tables, self.n_bits, d))

    def _hash_one(self, x):
        assert self.proj_ is not None
        buckets = []
        for t in range(self.n_tables):
            z = self.proj_[t] @ x
            bits = (z >= 0).astype(np.uint8)
            code = 0
            for k, b in enumerate(bits):
                if b:
                    code |= 1 << k
            buckets.append((t, int(code)))
        return buckets

    def _pairs(self, buckets):
        for u, v in combinations(buckets, 2):
            yield u if u < v else v, v if u < v else u

    def _ensure_class(self, c):
        if c not in self.W_:
            self.W_[c] = defaultdict(float)
        if self.use_unary and c not in self.B_:
            self.B_[c] = defaultdict(float)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if self.proj_ is None:
            self._build_proj(X.shape[1])
        if self.classes_ is None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, np.unique(y)]))

        eta, lam = self.learning_rate, self.decay
        for xi, yi in zip(X, y):
            yi = int(yi)
            self._ensure_class(yi)
            buckets = self._hash_one(xi)

            if lam > 0.0:
                for p in self._pairs(buckets):
                    self.W_[yi][p] *= 1.0 - lam
                if self.use_unary:
                    for u in buckets:
                        self.B_[yi][u] *= 1.0 - lam

            for p in self._pairs(buckets):
                self.W_[yi][p] += eta
            if self.use_unary:
                for u in buckets:
                    self.B_[yi][u] += eta
        return self

    def decision_function(self, X):
        C = len(self.classes_)
        scores = np.zeros((X.shape[0], C), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i, xi in enumerate(X):
            buckets = self._hash_one(xi)
            for u, v in self._pairs(buckets):
                for c, Wc in self.W_.items():
                    scores[i, class_to_idx[c]] += Wc.get((u, v), 0.0)
            if self.use_unary:
                for u in buckets:
                    for c, Bc in self.B_.items():
                        scores[i, class_to_idx[c]] += self.unary_weight * Bc.get(u, 0.0)
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[scores.argmax(axis=1)]
