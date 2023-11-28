import superfastcode
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import multiprocessing
from scipy import stats as st
from copy import deepcopy as copy
from joblib import delayed, Parallel
from pathos.multiprocessing import ProcessPool
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
import time

class FairRandomForestClassifier():
    def __init__(
            self,
            n_jobs=-1,
            n_bins=256,
            max_depth=7,
            bootstrap=True,
            random_state=42,
            n_estimators=500,
            orthogonality=.5,
            min_samples_leaf=3,
            min_samples_split=7,
            max_features="auto",
            sampling_proportion=1.0,
            hash_values=True
            # the estimate proportion of unique samples
            # equivalent to sampling_proportion=1, bootstrap=True
            # https://stats.stackexchange.com/questions/126107/
    ):
        """
        Fair Random Forest Classifier
        n_estimators -> int:
            number of FairDecisionTreeClassifier objects
        n_bins -> int:
            feature quantiles from which candidate splits are generated
        min_samples_split -> int:
            smallest number of samples in a node for a split to be considered
        min_samples_leaf -> int:
            smallest number of samples in each leaf after a split for that split to be considered
        max_depth -> int:
            max number of allowed splits per tree
        sampling_proportion -> float:
            proportion of samples to resample in each tree
        max_features -> int:
            number of samples to bootstrap
                     -> float:
            proportion of samples to bootstrap
                     -> str:
            "auto"/"sqrt": sqrt of the number of features is used
            "log"/"log2": log2 of the number of features is used
        bootstrap -> bool:
            bootstrap strategy with (True) or without (False) replacement
        random_state -> int:
            seed for all random processes
        criterion -> str:
            score criterion for splitting
            {"scaff", "kamiran"}
        kamiran_method -> str
            operation to combine IGC and IGS when criterion=='kamiran'
            {"sum", "sub", "div"}
        split_info_norm -> str
            denominator in gain normalisation:
            {"entropy", "entropy_inv", None}
        oob_pruning -> bool
            if out of bag samples (when sample_proportion!=1.0 or bootstrap==True) should be used to prune after fitting
        orthogonality -> int/float:
            strength of fairness constraint in which:
            0 is no fairness constraint (i.e., 'traditional' classifier)
            [0,1]
        n_jobs -> int:
            CPU usage; -1 for all threads
        """

        self.is_fit = False
        self.n_jobs = n_jobs
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.sampling_proportion = sampling_proportion
        self.hash_values = hash_values


    def fit(self, X="X", y="y", s="s", **kwargs):
        """
        X -> any_dim pandas.df or np.array: numerical/categorical
        y -> one_dim pandas.df or np.array: only binary
        s -> any_dim pandas.df or np.array: columns must be binary
        """

        # Defining make_batches inside fit() and predict_proba() leads to duplicate code
        # but also significantly reduced runtime
        def make_batches():
            n_jobs = self.n_jobs
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            len_iterable = len(self.trees)
            if len_iterable < n_jobs:
                n_jobs = len_iterable
            batches = [[] for i in range(n_jobs)]
            for i in range(len_iterable):
                batches[i % n_jobs].append(self.trees[i])
            return batches

        def fit_batch(batch_trees, X, y, s):
            for tree in batch_trees:
                tree.fit(X, y, s)
            return batch_trees

        # Generating FairRandomForest

        np.random.seed(self.random_state)
        # this is the range of all possible seed values in numpy
        random_states = np.random.randint(0, 2 ** 31, self.n_estimators)
        while len(np.unique(random_states)) != len(random_states):
            random_states = np.random.randint(0, 2 ** 31, self.n_estimators)

        self.trees = [
            superfastcode.FDTC(
                    n_bins=self.n_bins,
                    max_depth=self.max_depth,
                    bootstrap=self.bootstrap,
                    random_state=random_state,
                    orthogonality=self.orthogonality,
                    max_features=self.max_features,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    sampling_proportion=self.sampling_proportion,
                    hash_values=self.hash_values
            )
            for random_state in random_states
        ]
        self.classes_ = np.unique(y)
        self.pred_th = (y == 1).sum() / len(y)
        self.s = np.array(s).astype(object) if (
                "fit_params" not in list(kwargs.keys())
        ) else (
            np.array(kwargs["fit_params"]["s"]).astype(object)
        )

        if self.n_estimators == 1:
            self.n_jobs = 1
        if self.n_jobs == 1:
            for tree in self.trees:
                tree.fit(X, y, s)
        else:
            batches_trees = make_batches()

            fit_batches_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_batch)(
                    batch_trees,
                    X,
                    y,
                    s,
                ) for batch_trees in batches_trees
            )
            self.trees = [tree for fit_batch_trees in fit_batches_trees for tree in fit_batch_trees]

        self.is_fit = True

    def predict_proba(self, X, mean_type="prob"):
        """
        Retuns the predicted probabilties of input X
        mean_type -> str
            Method to compute the probailities across all trees
            {"prob", "pred"}
            "prob" computes the mean of all tree probabilities (the probability of Y=1 of each terminal node)
            "pred" computes the mean of all tree predicitons {0, 1}
        """

        def make_batches():
            n_jobs = self.n_jobs
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            len_trees = len(self.trees)
            if len_trees < n_jobs:
                n_jobs = len_trees
            batches = [[] for i in range(n_jobs)]
            for i in range(len_trees):
                item = self.trees[i]
                batches[i % n_jobs].append(item)
            return batches

        def predict_proba_batch(batch_trees, X, mean_type):
            batch_prob = []
            if mean_type == "prob":
                for tree in batch_trees:
                    batch_prob.append(np.array(tree.predict_proba(X))[:, 1])
            elif mean_type == "pred":
                for tree in batch_trees:
                    batch_prob.append(np.array(tree.predict(X)))
            return batch_prob

        if self.n_jobs == 1:
            if mean_type == "prob":
                y_prob = np.mean(
                    [np.array(tree.predict_proba(X))[:, 1] for tree in self.trees],
                    axis=0
                ).reshape(-1, 1)

            elif mean_type == "pred":
                y_prob = np.mean(
                    [np.array(tree.predict(X)) for tree in self.trees],
                    axis=0
                ).reshape(-1, 1)

        else:
            batches_trees = make_batches()
            proba_batches = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_proba_batch)(
                    batch_trees,
                    X,
                    mean_type
                ) for batch_trees in batches_trees
            )
            y_prob = np.mean(
                [prob for proba_batch in proba_batches for prob in proba_batch],
                axis=0
            ).reshape(-1, 1)

        return np.concatenate(
            (1 - y_prob, y_prob),
            axis=1
        )

    def predict(self, X, mean_type="prob"):
        """
        Retuns the predicted class label of input X
        theta -> float
            orthogonality parameter for the kamiran method when criterion=="kamiran"
            if not specified, the orthogonality parameter given in init is used instead
        mean_type -> str
            {"prob", "pred"}
            Method to compute the probailities across all trees, with which the np.mean([0.5, self.pred_th]) is the threshold
            Note: self.pred_th is given as the proportion of positive class instances [P(Y=1)]
            "prob" computes the mean of all tree probabilities (the probability of Y=1 of each terminal node)
            "pred" computes the mean of all tree predicitons {0, 1}
        """
        return (self.predict_proba(X, mean_type)[:, 1] >= np.mean([0.5, self.pred_th])).astype(int)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=False):
        if deep:
            return copy({
                "n_jobs": self.n_jobs,
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
                "n_estimators": self.n_estimators,
                "orthogonality": self.orthogonality,
                "criterion": self.criterion,
                "oob_pruning": self.oob_pruning,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "max_features": self.max_features,
                "split_info_norm": self.split_info_norm,
                "sampling_proportion": self.sampling_proportion,
            })
        else:
            return {
                "n_jobs": self.n_jobs,
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
                "n_estimators": self.n_estimators,
                "orthogonality": self.orthogonality,
                "criterion": self.criterion,
                "oob_pruning": self.oob_pruning,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "max_features": self.max_features,
                "split_info_norm": self.split_info_norm,
                "sampling_proportion": self.sampling_proportion,
            }

    def __str__(self):
        string = "FairDecisionTreeClassifier():" + "\n" + \
                 "  is_fit=" + str(self.is_fit) + "\n" + \
                 "  n_bins=" + str(self.n_bins) + "\n" + \
                 "  max_depth=" + str(self.max_depth) + "\n" + \
                 "  criterion=" + str(self.criterion) + "\n" + \
                 "  bootstrap=" + str(self.bootstrap) + "\n" + \
                 "  max_features=" + str(self.max_features) + "\n" + \
                 "  random_state=" + str(self.random_state) + "\n" + \
                 "  n_estimators=" + str(self.n_estimators) + "\n" + \
                 "  orthogonality=" + str(self.orthogonality) + "\n" + \
                 "  oob_pruning=" + str(self.oob_pruning) + "\n" + \
                 "  split_info_norm=" + str(self.split_info_norm) + "\n" + \
                 "  min_samples_leaf=" + str(self.min_samples_leaf) + "\n" + \
                 "  min_samples_split=" + str(self.min_samples_split) + "\n" + \
                 "  sampling_proportion=" + str(self.sampling_proportion)

        return string

    def __repr__(self):
        return self.__str__()

    def to_list(self):
        if not self.is_fit:
            raise RuntimeError("No forest to convert to list; forest has not been fit yet")
        return [tree.to_list() for tree in self.trees]
