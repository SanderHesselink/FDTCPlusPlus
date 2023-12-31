import numpy as np
import pandas as pd
import multiprocessing
from scipy import stats as st
from copy import deepcopy as copy
from joblib import delayed, Parallel
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
import time


class FairDecisionTreeClassifier():
    def __init__(
            self,
            n_bins=256,
            max_depth=7,
            bootstrap=False,
            random_state=42,
            orthogonality=.5,
            max_features=1.0,
            # Not necessary
            oob_pruning=True,
            # Always scaff
            criterion="scaff",
            min_samples_leaf=3,
            min_samples_split=7,
            # Not relevant
            kamiran_method=None,
            # Not relevant
            split_info_norm="entropy",
            sampling_proportion=1.0
    ):
        self.is_fit = False
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.oob_pruning = oob_pruning
        self.split_info_norm = split_info_norm
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.sampling_proportion = sampling_proportion
        self.n_bins = np.inf if (n_bins is None) else n_bins
        self.max_depth = np.inf if (max_depth is None) else (max_depth)
        self.kamiran_method = "sum" if (kamiran_method is None) else (kamiran_method)

    def fit(self, X="X", y="y", s="s", **kwargs):
        """
        X -> pandas.df: may contain int float str
        y -> one_dim pandas.df or np.array: only binary int {0, 1}
        s -> any_dim pandas.df or np.array: only str
        kwargs -> for compatibility with scikit-learn: fit_params in cross_validate()
        """

        # self.X_source = copy(X)
        # we use pandas to sort out between the numerical and categorical variables
        if "pandas" not in str(type(X)):
            X = pd.DataFrame(X)

        self.X = X
        self.y = np.array(y).astype(int)

        # for compatibility with scikit-learn since sklearn fit() methods only take X, y
        self.s = np.array(s).astype(str) if (
                "fit_params" not in list(kwargs.keys())
        ) else (
            np.array(kwargs["fit_params"]["s"]).astype(str)
        )

        if (len(self.X) != len(self.y)) or (len(self.X) != len(self.s)) or (len(self.y) != len(self.s)):
            raise Exception("X, y, and s lenghts do not match")
        if len(self.y.shape) == 1 or len(self.y.ravel()) == len(self.X):
            self.y = self.y.ravel()
        if len(self.s.shape) == 1 or len(self.s.ravel()) == len(self.X):
            self.s = self.s.reshape(-1, 1)


        np.random.seed(self.random_state)
        if (self.sampling_proportion != 1.0) or (self.bootstrap):
            indexs_to_keep = []
            # ensuring sampling is stratified
            split_groups = (
                pd.DataFrame(self.s).apply(lambda x: "_".join(x), axis=1).astype(str) + "_" + pd.Series(self.y).astype(
                    str) if (
                        self.s.shape[1] > 1
                ) else (
                        pd.Series(self.s.ravel()).astype(str) + "_" + pd.Series(self.y).astype(str)
                )
            )
            all_indexs = np.array(range(len(self.X)))
            for split_group in np.unique(split_groups):
                indexs = all_indexs[(split_groups == split_group).values].copy()
                sampling_n = max(1, int(round(len(indexs) * self.sampling_proportion)))
                indexs_to_keep += np.random.choice(
                    indexs,
                    size=sampling_n,
                    replace=self.bootstrap
                ).tolist()
            indexs_to_keep = np.array(indexs_to_keep)
            self.X = self.X.iloc[indexs_to_keep]
            self.y = self.y[indexs_to_keep]
            self.s = self.s[indexs_to_keep]

        # computing once
        self.y_pos_bool = self.y == 1
        self.y_neg_bool = ~self.y_pos_bool
        self.s_bool_dict = {}
        for s_column in range(self.s.shape[1]):
            self.s_bool_dict[s_column] = {}
            unique_s = np.unique(self.s[:, s_column])
            for s in unique_s:
                self.s_bool_dict[s_column][s] = (self.s[:, s_column]) == s
        # for binary-unicategorical methods
        self.s_pos_bool = self.s_bool_dict[0][self.s[0, 0]]
        self.s_neg_bool = ~self.s_pos_bool
        self.s_pos_bool
        self.s_neg_bool
        # for compatibility with scikit-learn
        self.classes_ = np.unique(y)
        


        # feature sampling
        if "int" in str(type(self.max_features)):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, self.max_features),
                replace=False
            ))
        elif "float" in str(type(self.max_features)):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, int(round(self.X.shape[1] * self.max_features))),
                replace=False
            ))
        elif ("auto" in str(self.max_features)) or ("sqrt" in str(self.max_features)):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, int(round(np.sqrt(self.X.shape[1])))),
                replace=False
            ))
        elif "log" in str(self.max_features):
            self.features = sorted(np.random.choice(
                self.X.columns,
                size=max(1, int(round(np.log2(self.X.shape[1])))),
                replace=False
            ))
        else:
            self.features = self.X.columns
        self.X = self.X[self.features]

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        categorical_part = self.X.select_dtypes(exclude=numerics)
        numeric_part = self.X.select_dtypes(include=numerics)
        # OHE to be applied to X from predict method
        self.ohe = OneHotEncoder(handle_unknown="ignore").fit(categorical_part)
        # X is now an array
        self.X = np.concatenate(
            (
                numeric_part.values.astype(float),
                self.ohe.transform(categorical_part).toarray()
            ), axis=1
        )


        self.feature_value_idx_bool = {}
        for feature in range(self.X.shape[1]):
            self.feature_value_idx_bool[feature] = {}
            # discarding the first value since there is no "less than lowest"
            unique_values = np.unique(self.X[:, feature])
            if len(unique_values) >= 2:
                # if the number of split values is "too large"
                if len(unique_values) > (self.n_bins):
                    unique_values = np.unique(np.quantile(
                        a=unique_values,
                        q=np.linspace(0, 1, self.n_bins),
                        method="nearest"
                    ))
                for value in unique_values[1:]:
                    idx_bool = (self.X[:, feature] < value)
                    self.feature_value_idx_bool[feature][value] = idx_bool

        # prediction threshold
        self.pred_th = self.y.sum() / len(self.y)
        self.indexs = np.repeat(True, len(self.X))

        if self.criterion == "scaff":
            scaff_parent = (1 - self.orthogonality) * 0.5 - self.orthogonality * 0.5

            def evaluate_split(feature, value, indexs):
                left_bool = self.feature_value_idx_bool[feature][value] & indexs
                right_bool = (~self.feature_value_idx_bool[feature][value]) & indexs
                # if split results in 2 non-empty partitions with min samples leaf size
                if (left_bool.sum() >= self.min_samples_leaf) and (right_bool.sum() >= self.min_samples_leaf):
                    # focusing on either left or right bool is fine as long as we take the max auc
                    # auc_y
                    tpr_y = ((self.y_pos_bool) & left_bool).sum() / ((self.y_pos_bool) & indexs).sum()
                    fpr_y = ((self.y_neg_bool) & left_bool).sum() / ((self.y_neg_bool) & indexs).sum()
                    auc_y = (1 + tpr_y - fpr_y) / 2
                    auc_y = max(auc_y, 1 - auc_y)

                    # auc_s
                    auc_s_list = []
                    for s_column in range(self.s.shape[1]):
                        unique_s = np.unique(self.s[indexs, s_column])
                        # if more than 1 sensitive attribute is present
                        if len(unique_s) >= 2:
                            for s in unique_s:
                                s_pos = self.s_bool_dict[s_column][s]
                                s_neg = ~s_pos
                                tpr_s = (s_pos & left_bool).sum() / (s_pos & indexs).sum()
                                fpr_s = (s_neg & left_bool).sum() / (s_neg & indexs).sum()
                                auc_s = (1 + tpr_s - fpr_s) / 2
                                auc_s = max(auc_s, 1 - auc_s)
                                auc_s_list.append(auc_s)
                                if len(unique_s) == 2:
                                    break
                        else:
                            auc_s = 1
                            auc_s_list.append(auc_s)
                            break
                    auc_s = max(auc_s_list)

                    scaff_child = (1 - self.orthogonality) * auc_y - self.orthogonality * auc_s
                    scaff_gain = scaff_child - scaff_parent
                    split_info = st.entropy([left_bool.sum(), right_bool.sum()], base=2)

                    score = scaff_gain / split_info
                else:
                    score = -np.inf


                return score


        def get_best_split(indexs):
            best_score = 0
            best_value = np.nan
            best_feature = np.nan
            for feature in range(self.X.shape[1]):
                unique_values = np.unique(self.X[indexs, feature])
                if len(unique_values) >= 2:
                    unique_intersect = np.intersect1d(
                        unique_values[1:],
                        np.array(list(self.feature_value_idx_bool[feature].keys()))
                    )
                    # we know that the unique_values[0] is no-good as splitter
                    # it would generate a left empty node, and a right full node
                    for value in unique_intersect:
                        split_score = evaluate_split(feature, value, indexs)
                        if split_score >= best_score:
                            best_score = split_score
                            best_feature = feature
                            best_value = value
            #print(best_score, best_feature, best_value)
            return best_score, best_feature, best_value

        # recursively grow the actual tree
        def build_tree(indexs, depth=0):
            tree = {}
            if (
                    len(np.unique(self.y[indexs])) == 1 or (  # no need to split if there is already only 1 y class
                    indexs.sum() < self.min_samples_split) or (  # minimum number of samples to consider a split
                    depth == self.max_depth)  # if we've reached the max depth in the tree
            ):
                class_prob = (self.y_pos_bool & indexs).sum() / indexs.sum()
                if self.criterion == "scaff":
                    return class_prob


            else:
                score, feature, value = get_best_split(indexs)
                if np.isnan(feature):  ## in case no more feature values exist for splitting
                    class_prob = (self.y_pos_bool & indexs).sum() / indexs.sum()
                    if self.criterion == "scaff":
                        return class_prob


                else:
                    left_indexs = self.feature_value_idx_bool[feature][value] & indexs
                    right_indexs = (~self.feature_value_idx_bool[feature][value]) & indexs

                    tree[(feature, value)] = {
                        "prob": (self.y[indexs]).sum() / indexs.sum(),
                        "<": build_tree(left_indexs, depth=depth + 1),
                        ">=": build_tree(right_indexs, depth=depth + 1),
                    }
                    return tree

        self.tree = build_tree(self.indexs)
        self.is_fit = True

    def predict_proba(self, X, theta=None):
        if "pandas" not in str(type(X)):
            X = pd.DataFrame(X)

        X = X[self.features]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        categorical_part = X.select_dtypes(exclude=numerics)
        numeric_part = X.select_dtypes(include=numerics)
        # X is now an array
        X = np.concatenate(
            (
                numeric_part.values.astype(float),
                self.ohe.transform(categorical_part).toarray()
            ), axis=1
        )

        if self.criterion == "scaff":
            def get_prob(X, tree, y_prob=None, idx_bool=None):
                y_prob = np.repeat(np.nan, len(X)) if (y_prob is None) else (y_prob)
                idx_bool = np.repeat(True, len(X)) if (idx_bool is None) else (idx_bool)
                if type(tree) == type({}):
                    feature, value = list(tree.keys())[0]
                    left_bool = (X[:, feature] < value) & idx_bool
                    right_bool = (X[:, feature] >= value) & idx_bool
                    sub_tree_left = tree[(feature, value)]["<"]
                    sub_tree_right = tree[(feature, value)][">="]
                    y_prob = get_prob(X, sub_tree_left, y_prob, left_bool)
                    y_prob = get_prob(X, sub_tree_right, y_prob, right_bool)
                    return y_prob

                else:
                    y_prob[idx_bool] = tree
                    return y_prob

            y_prob = get_prob(X, self.tree).reshape(-1, 1)


        return np.concatenate(
            ((1 - y_prob), y_prob),
            axis=1
        )

    def predict(self, X, theta=None):
        y_prob = self.predict_proba(X, theta)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        return y_pred

    # for compatibility with scikit-learn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # for compatibility with scikit-learn
    def get_params(self, deep=False):
        if deep:
            return copy({
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "criterion": self.criterion,
                "bootstrap": self.bootstrap,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "orthogonality": self.orthogonality,
                "oob_pruning": self.oob_pruning,
                "kamiran_method": self.kamiran_method,
                "split_info_norm": self.split_info_norm,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "sampling_proportion": self.sampling_proportion,
            })

        else:
            return {
                "n_bins": self.n_bins,
                "max_depth": self.max_depth,
                "criterion": self.criterion,
                "bootstrap": self.bootstrap,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "orthogonality": self.orthogonality,
                "oob_pruning": self.oob_pruning,
                "kamiran_method": self.kamiran_method,
                "split_info_norm": self.split_info_norm,
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
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
                 "  orthogonality=" + str(self.orthogonality) + "\n" + \
                 "  oob_pruning=" + str(self.oob_pruning) + "\n" + \
                 "  kamiran_method=" + str(self.kamiran_method) + "\n" + \
                 "  split_info_norm=" + str(self.split_info_norm) + "\n" + \
                 "  min_samples_leaf=" + str(self.min_samples_leaf) + "\n" + \
                 "  min_samples_split=" + str(self.min_samples_split) + "\n" + \
                 "  sampling_proportion=" + str(self.sampling_proportion)

        return string

    def __repr__(self):
        return self.__str__()


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
            oob_pruning=True,
            criterion="scaff",
            min_samples_leaf=3,
            min_samples_split=7,
            max_features="auto",
            kamiran_method=None,
            split_info_norm=None,
            sampling_proportion=1.0,
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
        self.criterion = criterion #Not relevant
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.orthogonality = orthogonality
        self.oob_pruning = oob_pruning #Not relevant
        self.kamiran_method = kamiran_method #Not relevant
        self.split_info_norm = split_info_norm #Not relevant
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.sampling_proportion = sampling_proportion

    def fit(self, X="X", y="y", s="s", **kwargs):
        """
        X -> any_dim pandas.df or np.array: numerical/categorical
        y -> one_dim pandas.df or np.array: only binary
        s -> any_dim pandas.df or np.array: columns must be binary
        """

        def make_batches(iterable, n_jobs=-1):
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            len_iterable = len(iterable)
            if len_iterable < n_jobs:
                n_jobs = len_iterable
            batches = [[] for i in range(n_jobs)]
            for i in range(len_iterable):
                item = iterable[i]
                batches[i % n_jobs].append(item)
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

        trees = [
            FairDecisionTreeClassifier(
                n_bins=self.n_bins,
                max_depth=self.max_depth,
                bootstrap=self.bootstrap,
                criterion=self.criterion,
                random_state=random_state,
                max_features=self.max_features,
                orthogonality=self.orthogonality,
                kamiran_method=self.kamiran_method,
                oob_pruning=self.oob_pruning,
                split_info_norm=self.split_info_norm,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                sampling_proportion=self.sampling_proportion,
            )
            for random_state in random_states
        ]

        self.trees = trees
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
            batches_trees = make_batches(self.trees, n_jobs=self.n_jobs)
            fit_batches_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_batch)(
                    batch_trees,
                    X,
                    y,
                    s,
                ) for batch_trees in batches_trees
            )
            self.trees = [tree for fit_batch_trees in fit_batches_trees for tree in fit_batch_trees]

        self.fit = True

    def predict_proba(self, X, theta=None, mean_type="prob"):
        """
        Retuns the predicted probabilties of input X
        theta -> float
            orthogonality parameter for kamiran
            if not specified, the orthogonality parameter given in init is used instead
        mean_type -> str
            Method to compute the probailities across all trees
            {"prob", "pred"}
            "prob" computes the mean of all tree probabilities (the probability of Y=1 of each terminal node)
            "pred" computes the mean of all tree predicitons {0, 1}
        """

        def make_batches(iterable, n_jobs=-1):
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            len_iterable = len(iterable)
            if len_iterable < n_jobs:
                n_jobs = len_iterable
            batches = [[] for i in range(n_jobs)]
            for i in range(len_iterable):
                item = iterable[i]
                batches[i % n_jobs].append(item)
            return batches

        def predict_proba_batch(batch_trees, X, theta, mean_type):
            batch_prob = []
            if mean_type == "prob":
                for tree in batch_trees:
                    batch_prob.append(tree.predict_proba(X, theta=theta)[:, 1])
            elif mean_type == "pred":
                for tree in batch_trees:
                    batch_prob.append(tree.predict(X, theta=theta))
            return batch_prob

        if self.n_jobs == 1:
            if mean_type == "prob":
                y_prob = np.mean(
                    [tree.predict_proba(X, theta=theta)[:, 1] for tree in self.trees],
                    axis=0
                ).reshape(-1, 1)

            elif mean_type == "pred":
                y_prob = np.mean(
                    [tree.predict(X, theta=theta) for tree in self.trees],
                    axis=0
                ).reshape(-1, 1)

        else:
            batches_trees = make_batches(self.trees, n_jobs=self.n_jobs)
            proba_batches = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_proba_batch)(
                    batch_trees,
                    X,
                    theta,
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

    def predict(self, X, theta=None, mean_type="prob"):
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
        return (self.predict_proba(X, theta, mean_type)[:, 1] >= np.mean([0.5, self.pred_th])).astype(int)

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
