#include <Windows.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <variant>
#include <tuple>
#include <valarray>
#include <queue>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <numeric>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <functional>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

namespace py = pybind11;
using namespace std;

// helper functions

// Valarray's sum() method uses operator+=
// For boolean values, this operator behaves like a logical OR
// If we want the amount of true values in the valarray, we need to use count.
// 
// This helper function exists so we only have to slice once
// i.e. boolsum(valarray(slice)) instead of
// count(begin(valarray(slice)), end(valarray(slice))), which is doing the same slice operation twice
inline int boolsum(const valarray<bool> & a) {
    return count(begin(a), end(a), true);
}

// find unique values in valarray
template <class T>
inline vector<T> unique_valarray(const valarray<T>& a) {
    // Using the default constructor of a set is faster than
    // turning the valarray into vector, sorting the vector,
    // and then using std::unique
    //
    // Using an unordered set would probably be even faster,
    // but unfortunately, we require the output to be sorted
    const auto ret = set<T>(begin(a), end(a));
    // We prefer returning a vector over a set or valarray for portability
    return vector<T>(ret.begin(), ret.end());
}


struct TreeNode {
    int feature{ -1 };
    variant<float, double, int, bool, string> value{ NULL };
    float prob{ NAN };
    struct shared_ptr<TreeNode> less{ nullptr };
    struct shared_ptr<TreeNode> more_or_equal{ nullptr };

    TreeNode() {}
    // Constructor in case of leaf
    TreeNode(float prob_) : prob(prob_) {}
    TreeNode(
        int feature_,
        variant<float, double, int, bool, string> value_,
        float prob_,
        shared_ptr<TreeNode> & less_,
        shared_ptr<TreeNode> & more_or_equal_
    ) :
        feature(feature_),
        value(value_),
        prob(prob_),
        less(less_),
        more_or_equal(more_or_equal_)
    {}
};

template <class T>
class DataMatrix {
public:
    vector<size_t> shape;
    DataMatrix() {}
    // We know we only need to construct this DataMatrix from a vector of vectors, 
    // after it has already been checked for empty, etc. Generalization would require 
    // exception handling and additional constructors, but is currently just not necessary.
    DataMatrix(const vector<vector<T>> & data_) 
    {   
        n_rows = data_.size();
        n_cols = data_[0].size();
        shape = { n_rows, n_cols };
        data = valarray<T>(n_cols * n_rows);
        for (int x = 0; x < n_rows; ++x) {
            for (int y = 0; y < n_cols; ++y) {
                data[x * n_cols + y] = data_[x][y];
            }
        }
    }

    // Because data access only happens in ranged loops, testing should eliminate the 
    // need for exception handling.
    inline T& operator() (const int x, const int y) {
        return data[x * n_cols + y];
    }
    inline T operator() (const int x, const int y) const {
        return data[x * n_cols + y];
    }
    
    inline valarray<T> operator() (const int row) {
        return data[slice(row * n_cols, n_cols, 1)];
    }
    inline valarray<T> operator() (const int row) const {
        return data[slice(row * n_cols, n_cols, 1)];
    }
    
    // In order to differentiate between slicing a row or a column,
    // the column slice operator takes an additional char as parameter.
    // In this code, we always use 'c' (for "column")
    inline valarray<T> operator() (const char, const int column) {
        return data[slice(column, n_rows, n_cols)];
    }
    inline valarray<T> operator() (const char, const int column) const {
        return data[slice(column, n_rows, n_cols)];
    }


private:
    valarray<T> data;
    size_t n_rows;
    size_t n_cols;
};

// treelist and unpack_treelist are mainly used for pickling the FairDecisionTreeClassifier object,
// but treelist is also exposed via pybind11 to create a representation of the tree in Python.
vector<tuple<int, variant<float, double, int, bool, string>, float>> treelist(TreeNode& t, vector<tuple<int, variant<float, double, int, bool, string>, float>>& list) {
    list.push_back({ t.feature, t.value, t.prob });
    if (t.feature != -1) {
        treelist(*t.less, list);
        treelist(*t.more_or_equal, list);
    }

    return list;
}

// Recursively turning a treelist into a tree
TreeNode unpack_treelist(vector<tuple<int, variant<float, double, int, bool, string>, float>>& list) {
    auto [feature, value, prob] = list[0];
    list.erase(list.begin());
    if (feature != -1) {
        shared_ptr<TreeNode> leftnode = make_shared<TreeNode>(unpack_treelist(list));
        shared_ptr<TreeNode> rightnode = make_shared<TreeNode>(unpack_treelist(list));
        auto t = TreeNode(
            feature,
            value,
            prob,
            leftnode,
            rightnode
        );

        return t;
    }
    else {
        return TreeNode(prob);
    }
}

class FairDecisionTreeClassifier {
private:
    int n_bins;
    int max_depth;
    bool bootstrap;
    int random_state;
    float orthogonality;
    variant<float, string> max_features;
    int min_samples_leaf;
    int min_samples_split;
    float sampling_proportion;
    bool hash_values;

    // The python code stores variables like X and y as members,
    // but this is not necessary and not very memory efficient.
    // Making them local variables does mean function calls become a bit more complicated.


    float evaluate_split(
        const int feature, 
        const variant<float, double, int, bool, string> value,
        const valarray<bool> & indexmask,
        const valarray<bool> & y_pos,
        const unordered_map<int, map<variant<float, double, int, bool, string>, valarray<bool>>>& feature_value_idx_bool,
        const DataMatrix<size_t> & s,
        const unordered_map<int, unordered_map<size_t, valarray<bool>>> & s_bool_map)
    {
        // focusing on either left or right bool is fine as long as we take the max auc, 
        // testing shows that always slecting the most sparse side has 0 effect
        const auto left_bool = feature_value_idx_bool.at(feature).at(value) & indexmask;
        const auto lsize = boolsum(left_bool);
        const auto isize = boolsum(indexmask);

        // If split results in 2 non-empty partitions with min samples leaf size
        if (
            (lsize < min_samples_leaf) ||
            (isize - lsize) < min_samples_leaf) {
            return -1;
        }

        const float auc_y = [&]() {
            const float pos_sum = boolsum(y_pos & left_bool);
            const float neg_sum = lsize - pos_sum;
            const float total_pos = boolsum(y_pos & indexmask);
            const float total_neg = isize - total_pos;

            const float tpr_y = pos_sum / total_pos;
            const float fpr_y = neg_sum / total_neg;

            const float auc = ((1.0 + tpr_y - fpr_y) / 2.0);
            return max(auc, 1.0 - auc);
        }();


        const float auc_s = [&]() {
            vector<float> auc_s_list;
            auc_s_list.reserve(s.shape[0]);
            for (int s_column = 0; s_column < s.shape[1]; ++s_column) {
                valarray<size_t> colslice = s('c', s_column)[indexmask];
                auto unique_s = unique_valarray(colslice);

                // If more than one sensitive attribute is present
                if (unique_s.size() >= 2) {
                    for (const auto & val : unique_s) {
                        const float auc_s_val = [&]() {
                            const valarray<bool> s_pos = s_bool_map.at(s_column).at(val);

                            const float pos_sum = boolsum(s_pos & left_bool);
                            const float neg_sum = lsize - pos_sum;
                            const float total_pos = boolsum(s_pos & indexmask);
                            const float total_neg = isize - total_pos;

                            const float tpr_s = pos_sum / total_pos;
                            const float fpr_s = neg_sum / total_neg;
                            const float auc = ((1.0 + tpr_s - fpr_s) / 2.0);
                            return max(auc, 1.0 - auc);
                        }();

                        auc_s_list.push_back(auc_s_val);
                        if (unique_s.size() == 2) {
                            break;
                        }
                    }
                }
                else {
                    return (float)1;
                }
            }
            return *max_element(auc_s_list.begin(), auc_s_list.end());
        }();

        const float scaff_gain = [=]() {
            const float scaff_parent = (1.0 - orthogonality) * 0.5 - orthogonality * 0.5;
            const float scaff_child = (1.0 - orthogonality) * auc_y - orthogonality * auc_s;
            return scaff_child - scaff_parent;
        }();
        
        
        // Calculate split info as entropy
        const float split_info = [=]() {
            const float left_sum = (float) lsize / (float) isize;
            const float right_sum = (float) (isize - (left_sum * isize)) / (float) isize;

            const float left_ent = left_sum * log(left_sum);
            const float right_ent = right_sum * log(right_sum);
            return -(left_ent + right_ent) / log(2.0);
        }();
        return scaff_gain / split_info;
    }

    tuple<float, int, variant<float, double, int, bool, string>> get_best_split(
        const valarray<bool> indexmask, 
        const DataMatrix<variant<float, double, int, bool, string>>& X, 
        const valarray<bool> & y_pos,
        const unordered_map<int, map<variant<float, double, int, bool, string>, valarray<bool>>>& feature_value_idx_bool,
        const DataMatrix<size_t> & s,
        const unordered_map<int, unordered_map<size_t, valarray<bool>>> & s_bool_map) {
        float best_score = 0;
        variant<float, double, int, bool, string> best_value = NAN;
        int best_feature = -1;
        for (int feature = 0; feature < X.shape[1]; ++feature) {
            const valarray<variant<float, double, int, bool, string>> colslice = X('c', feature)[indexmask];
            const auto unique_values = unique_valarray(colslice);

            if (unique_values.size() >= 2) {
                // Unique intersect in case of bins
                vector<variant<float, double, int, bool, string>> keylist;
                keylist.resize(feature_value_idx_bool.at(feature).size());
                vector<variant<float, double, int, bool, string>> unique_intersect;
                transform(
                    feature_value_idx_bool.at(feature).begin(),
                    feature_value_idx_bool.at(feature).end(),
                    keylist.begin(),
                    [](auto x) {return x.first;}
                );
                set_intersection(begin(unique_values), end(unique_values), keylist.begin(),
                    keylist.end(), back_inserter(unique_intersect));
                for (const auto & value : unique_intersect) {
                    const float split_score = evaluate_split(feature, value, indexmask, y_pos, feature_value_idx_bool, s, s_bool_map);
                    if (split_score >= best_score) {
                        best_score = split_score;
                        best_feature = feature;
                        best_value = value;
                    }
                }
            }
        }
        // Usefull prints for debugging
        //cout << best_score << " " << best_feature << " ";
        //visit([](auto x) { cout << x << endl;}, best_value);
        return { best_score, best_feature, best_value };
    }
    

    shared_ptr<TreeNode> build_tree(
        const valarray<bool> indexmask, 
        const DataMatrix<variant<float, double, int, bool, string>> & X, 
        const valarray<bool> & y_pos, 
        const unordered_map<int, map<variant<float, double, int, bool, string>, valarray<bool>>> & feature_value_idx_bool,
        const DataMatrix<size_t> & s,
        const unordered_map<int, unordered_map<size_t, valarray<bool>>> & s_bool_map,
        const int depth = 0) {
        const float class_prob = (float)boolsum(indexmask & y_pos) / (float)boolsum(indexmask);
        if (
            // No need to split if there is only one y class
            (!((y_pos & indexmask).sum())) ||
            (!((!y_pos & indexmask).sum())) ||
            // Minimum samples to consider a split
            (boolsum(indexmask) < min_samples_split) ||
            // Max depth has been reached
            (max_depth != -1 && depth == max_depth) 
            ) {
            return make_shared<TreeNode>(TreeNode(class_prob));
        }
        auto [score, feature, value] = get_best_split(indexmask, X, y_pos, feature_value_idx_bool, s, s_bool_map);
        if (feature == -1) {
            return make_shared<TreeNode>(TreeNode(class_prob));
        }
        else {
            const valarray<bool> left_index = feature_value_idx_bool.at(feature).at(value) & indexmask;
            const valarray<bool> right_index = !feature_value_idx_bool.at(feature).at(value) & indexmask;

            shared_ptr<TreeNode> leftnode = build_tree(left_index, X, y_pos, feature_value_idx_bool, s, s_bool_map, depth + 1);
            shared_ptr<TreeNode> rightnode = build_tree(right_index, X, y_pos, feature_value_idx_bool, s, s_bool_map, depth + 1);
            
            return make_shared<TreeNode>(TreeNode(feature, value, class_prob, leftnode, rightnode));
        }
    }

public:
    shared_ptr<TreeNode> tree;
    unordered_set<int> selected_features;
    map<int, bool> categorical;
    float pred_th;
    bool is_fit{ false };

    FairDecisionTreeClassifier(
        int n_bins_,
        int max_depth_,
        bool bootstrap_,
        int random_state_,
        float orthogonality_,
        variant<float, string> max_features_,
        int min_samples_leaf_,
        int min_samples_split_,
        float sampling_proportion_,
        bool hash_values_) :

        n_bins(n_bins_),
        max_depth(max_depth_),
        bootstrap(bootstrap_),
        random_state(random_state_),
        orthogonality(orthogonality_),
        max_features(max_features_),
        min_samples_leaf(min_samples_leaf_),
        min_samples_split(min_samples_split_),
        sampling_proportion(sampling_proportion_),
        hash_values(hash_values_)
    {};

    void fit(
        vector<vector<variant<float, double, int, bool, string>>> x, 
        vector<bool> y, 
        const vector<vector<string>> S,
        // For compatibility with scikit-learn since sklearn fit() methods only take X, y
        const vector<vector<string>> fit_params) 
    {
        vector<vector<string>> temps;
        if (!fit_params.empty()) {
            temps = fit_params;
        }
        else {
            temps = S;
        };

        if (x.size() == 0) {
            throw invalid_argument("Input is empty");
        }

        if (x.size() != y.size() || x.size() != temps.size() || y.size() != temps.size()) {
            throw invalid_argument("X, y, and s lenghts do not match");
        };
 
        if (x[0].size() == 0) {
            throw invalid_argument("X contains empty rows");
        }
        for (const auto & i : x) {
            if (i.size() != x[0].size()) {
                throw invalid_argument("X contains rows of varying lengths");
            }
        }
        const int ssize = temps[0].size();
        if (ssize == 0) {
            throw invalid_argument("s contains empty rows");
        }
        for (const auto & i : temps) {
            if (i.size() != ssize) {
                throw invalid_argument("s contains rows of varying lengths");
            }
        }
        // Stratified sampling
        // Current implementation is not pretty, but performance is not a bottleneck
        default_random_engine randomeng(random_state);
        auto indexmask_to_keep = valarray<bool>(true, y.size());
        if ((sampling_proportion != 1.0) || (bootstrap)) {
            auto indexes_to_keep = vector<int>();
            valarray<string> split_groups = valarray<string>(y.size());
            for (int i = 0; i < temps.size(); ++i) {
                string result = "";
                for (auto j : temps[i]) {
                    result += j + "_";
                }
                result += to_string(y[i]);
                split_groups[i] = result;
            }
            auto all_indexes = valarray<int>(y.size());
            iota(begin(all_indexes), end(all_indexes), 0);
            for (auto split_group : unique_valarray(split_groups)) {
                const valarray<int> indexes = all_indexes[split_groups == split_group];
                const int sampling_n = max(1, round(indexes.size() * sampling_proportion));
                // std::sample has no options for replacement, so bootstrap == true requires manual random sampling
                if (bootstrap) {
                    for (int i = 0; i < sampling_n; ++i) {
                        indexes_to_keep.push_back(indexes[randomeng() % indexes.size()]);
                    }
                }
                else {
                    sample(begin(indexes), end(indexes), back_inserter(indexes_to_keep), sampling_n, randomeng);            
                }
            }
            if (bootstrap) {
                x = [&]() {
                    vector<vector<variant<float, double, int, bool, string>>> temp;
                    for (auto i : indexes_to_keep) {
                        temp.push_back(x[i]);
                    }
                    return temp;
                }();
                y = [&]() {
                    vector<bool> temp;
                    for (auto i : indexes_to_keep) {
                        temp.push_back(y[i]);
                    }
                    return temp;
                }();
                temps = [&]() {
                    vector<vector<string>> temp;
                    for (auto i : indexes_to_keep) {
                        temp.push_back(temps[i]);
                    }
                    return temp;
                }();
            }
            else {
                indexmask_to_keep = !indexmask_to_keep;
                x = [&]() {
                    valarray<vector<variant<float, double, int, bool, string>>> temp = valarray<vector<variant<float, double, int, bool, string>>>(x.data(), x.size())[indexmask_to_keep];
                    return vector<vector<variant<float, double, int, bool, string>>>(begin(temp), end(temp));
                }();
                y = [&]() {
                    vector<bool> temp;
                    for (auto i : indexes_to_keep) {
                        temp.push_back(y[i]);
                    }
                    return temp;
                }();
                temps = [&]() {
                    valarray<vector<string>> temp = valarray<vector<string>>(temps.data(), temps.size())[indexmask_to_keep];
                    return vector<vector<string>>(begin(temp), end(temp));
                }();
            }
        }

        // Because operations on strings are very slow, and we don't care
        // about the exact value of s as long as we can differentiate categories,
        // we hash the values in s
        DataMatrix<size_t> s;
        hash<string> hasher;
        {
            auto temp_hash = vector<vector<size_t>>(temps.size());
            for (int i = 0; i < temps.size(); ++i) {
                temp_hash[i].resize(temps[0].size());
                for (int j = 0; j < temps[0].size(); ++j) {
                    temp_hash[i][j] = hasher(temps[i][j]);
                }
            }
            s = DataMatrix(temp_hash);
        }

        // Initiallize bool map
        unordered_map<int, unordered_map<size_t, valarray<bool>>> s_bool_map;
        for (int col = 0; col < s.shape[1]; ++col) {
            const auto unique_s = unique_valarray(s('c', col));
            for (const auto & cat : unique_s) {
                s_bool_map[col][cat] = s('c', col) == cat;
            }
        }

        // feature sampling
        // Again, not pretty, but not a bottleneck
        if (holds_alternative<float>(max_features)) {
            auto maxf = get<float>(max_features);
            if (maxf > 1 && maxf < x[0].size()) {
                while (selected_features.size() < maxf) {
                    selected_features.insert(randomeng() % x[0].size());
                }
            }
            else if (maxf < 1) {
                while (selected_features.size() < round(maxf * x[0].size())) {
                    selected_features.insert(randomeng() % x[0].size());
                }
            }
        }
        else if ((*get_if<string>(&max_features) == "auto") || (*get_if<string>(&max_features) == "sqrt")) {
            while (selected_features.size() < round(sqrt(x[0].size()))) {
                selected_features.insert(randomeng() % x[0].size());
            }
        }
        else if (*get_if<string>(&max_features) == "log") {
            while (selected_features.size() < round(log2(x[0].size()))) {
                selected_features.insert(randomeng() % x[0].size());
            }
        }
        if (!selected_features.empty()) {
            for (int i = 0; i < x.size(); ++i) {
                int j = 0;
                x[i].erase(remove_if(x[i].begin(), x[i].end(), [&](auto x) {
                    bool ret = find(selected_features.begin(), selected_features.end(), j) == selected_features.end();
                    ++j;
                    return ret;
                    }), x[i].end());
            }
        }


        for (int j = 0; j < x[0].size(); ++j) {
            if (holds_alternative<string>(x[0][j])) {
                categorical[j] = true;
                // We want to give the end user the option to turn of hashing
                // if explainability is important
                if (hash_values) {
                    for (int i = 0; i < x.size(); ++i) {
                        // A silent assumption here is that every column only holds
                        // a single datatype. If this is not the case,
                        // the code will not execute properly
                        x[i][j] = (int)hasher(get<string>(x[i][j]));
                            
                    }
                }
            }
            else {
                categorical[j] = false;
            }
        }

        auto X = DataMatrix(x);
        // vector<bool> has no method data(), so manual insertion is required
        // sigh
        valarray<bool> y_pos = valarray<bool>(X.shape[0]);
        for (int i = 0; i < X.shape[0]; ++i) {
            y_pos[i] = y[i];
        }
        y_pos = valarray<bool>(y_pos[indexmask_to_keep]);

        // initialize feature_value_idx_bool
        unordered_map<int, map<
            variant<float, double, int, bool, string>, valarray<bool>>
            > feature_value_idx_bool;
        for (int feature = 0; feature < X.shape[1]; ++feature) {
            const auto uniq_values = unique_valarray(X('c', feature));
            auto unique_values = valarray<variant<float, double, int, bool, string>>(uniq_values.data(), uniq_values.size());  


            if (unique_values.size() >= 2) {
                // If the number of split values is "too large"
                if (n_bins != -1 && unique_values.size() > n_bins) {
                    valarray<size_t> bindexes = valarray<size_t>(n_bins);
                    const float stepsize = (float)(unique_values.size()-1) / (float)(n_bins-1);
                    for (size_t i = 0; i < n_bins; ++i) {
                        bindexes[i] = round(i * stepsize);
                    }
                    unique_values = valarray<variant<float, double, int, bool, string>>(unique_values[bindexes]);
                }
                // We know that the unique_values[0] is no - good as splitter.
                // It would generate a left empty node, and a right full node.
                //
                // This is not the case if unique values is categorical, in which
                // case unique_values[0] still contains information.
                if (!categorical[feature]) {
                    unique_values = valarray<variant<float, double, int, bool, string>>(unique_values[slice(1, unique_values.size()-1, 1)]);
                }
                for (const auto & value : unique_values) {
                    feature_value_idx_bool[feature][value] = [&]() {
                        if (categorical[feature]) {
                            return X('c', feature) != value;
                        }
                        else {
                            return X('c', feature) < value;
                        }
                    }();
                }
            }
        }
        pred_th = (float)boolsum(y_pos) / (float)y.size();
        valarray<bool> indexmask = valarray<bool>(true, X.shape[0]);

        tree = build_tree(indexmask, X, y_pos, feature_value_idx_bool, s, s_bool_map);
        is_fit = true;
    }


    // Recursive probabilty prediction
    template <class T>
    valarray<float> get_prob(DataMatrix<T> & X, shared_ptr<TreeNode> node, valarray<float> y_prob, valarray<bool> indexmask) {
        auto feature = node->feature;
        auto value = node->value;
        if (feature == -1) {
            y_prob[indexmask] = node->prob;
            return y_prob;
        }
        if (categorical[feature]) {
            y_prob = get_prob(X, node->less, y_prob, (X('c', feature) != value) & indexmask);
            y_prob = get_prob(X, node->more_or_equal, y_prob, (X('c', feature) == value) & indexmask);
        }
        else {
            y_prob = get_prob(X, node->less, y_prob, (X('c', feature) < value) & indexmask);
            y_prob = get_prob(X, node->more_or_equal, y_prob, (X('c', feature) >= value) & indexmask);
        }
        return y_prob;
    }
 
    // Predict proba
    // Python version takes a Theta parameter, but this is not used in scaff prediction
    valarray<float> predict_proba(vector<vector<variant<float, double, int, bool, string>>> x) {
        if (!is_fit) {
            throw exception("Tree has not yet been fit");
        }

        // feature sampling
        if (!selected_features.empty()) {
            for (int i = 0; i < x.size(); ++i) {
                int j = 0;
                x[i].erase(remove_if(x[i].begin(), x[i].end(), [&](auto x) {
                    bool ret = find(selected_features.begin(), selected_features.end(), j) == selected_features.end();
                    ++j;
                    return ret;
                    }), x[i].end());
            }
        }

        if (x.size() == 0) {
            throw invalid_argument("Input is empty");
        }

        // Check if all rows of X and s are same length and rows aren't empty
        if (x[0].size() == 0) {
            throw invalid_argument("X contains empty rows");
        }
        for (const auto & i : x) {
            if (i.size() != x[0].size()) {
                throw invalid_argument("X contains rows of varying lengths");
            }
        }
        hash<string> hasher;
        for (int j = 0; j < x[0].size(); ++j) {
            // We assume the value of hash_values doesn't change between fitting and prediction
            // The only way it might change is through user input
            if (categorical[j] && hash_values) {
                for (int i = 0; i < x.size(); ++i) {
                    x[i][j] = (int)hasher(get<string>(x[i][j]));
                }
            }
        }
        auto X = DataMatrix(x);

        return get_prob(X, tree, valarray<float>(X.shape[0]), valarray<bool>(true, X.shape[0]));;
    }


    // Required output for predict_proba is a probability prediction for each class
    // i.e. a prediction of 0.3 leads to the output [0.7, 0.3]
    // (70% chance sample is 0, 30% chance sample is 1)
    vector<vector<float>> predict_proba_wrapper(vector<vector<variant<float, double, int, bool, string>>> x) {
        auto probs = predict_proba(x);
        vector<vector<float>> ret;
        ret.reserve(probs.size());
        for (int i = 0; i < probs.size(); ++i) {
            ret.push_back({ (float)1 - probs[i], probs[i] });
        }
        return ret;
    }

    valarray<bool> predict(vector<vector<variant<float, double, int, bool, string>>> & x) {
        return predict_proba(x) >= 0.5;
    }

    // get_params
    // Having a variant inside a variant is not great, but probably still faster than implementing a cast
    unordered_map<string, variant<float, double, int, bool, string, variant<float, string>>> get_params(bool deep) {
        unordered_map<string, variant<float, double, int, bool, string, variant<float, string>>> ret = {
            {"n_bins", n_bins},
            {"max_depth", max_depth},
            {"bootstrap", bootstrap},
            {"max_features", max_features},
            {"random_state", random_state},
            {"orthogonality", orthogonality},
            {"min_samples_leaf", min_samples_leaf},
            {"min_samples_split", min_samples_split},
            {"sampling_proportion", sampling_proportion},
            {"hash_values", hash_values}
        };
        return ret;
    }

    // set_params
    FairDecisionTreeClassifier& set_params(
        int n_bins_,
        int max_depth_,
        bool bootstrap_,
        variant<float, string> max_features_,
        int random_state_,
        float orthogonality_,
        int min_samples_leaf_,
        int min_samples_split_,
        float sampling_proportion_,
        bool hash_values_
        ) {
        // If (variable) would be cleaner than if (variable != NULL),
        // but since 0 and/or false might be accepted input, we have to be more precise
        if (n_bins_ != NULL) n_bins = n_bins_;
        if (max_depth_ != NULL) max_depth = max_depth_;
        if (bootstrap_ != NULL) bootstrap = bootstrap_;
        if ((get_if<float>(&max_features_) != NULL) || (get_if<string>(&max_features_) != NULL)) max_features = max_features_;
        if (random_state_ != NULL) random_state = random_state_;
        if (orthogonality_ != NULL) orthogonality = orthogonality_;
        if (min_samples_leaf_ != NULL) min_samples_leaf = min_samples_leaf_;
        if (min_samples_split_ != NULL) min_samples_split = min_samples_split_;
        if (sampling_proportion_ != NULL) sampling_proportion = sampling_proportion_;
        if (hash_values_ != NULL) hash_values = hash_values_;

        return *this;
    }

    string __str__() {
        stringstream ret;
        ret << "FairDecisionTreeClassifier():"
            << "\nis_fit = " << boolalpha << is_fit
            << "\nn_bins = " << n_bins
            << "\nmax_depth = " << max_depth
            << "\nbootstrap = " << bootstrap
            << "\nmax_features = ";
        visit([&](auto x) {ret << x;}, max_features);
        ret << "\nrandom_state = " << random_state
            << "\northogonality = " << orthogonality
            << "\nmin_samples_leaf = " << min_samples_leaf
            << "\nmin_samples_split = " << min_samples_split
            << "\nsampling_proportion = " << sampling_proportion
            << "\nhash_values = " << hash_values;

        return ret.str();
    }

    string __repr__() {
        return __str__();
    }

    // Calls the treelist helper function on the tree, meant for creating a representation of the tree in Python
    vector<tuple<int, variant<float, double, int, bool, string>, float>> to_list() {
        if (!is_fit) {
            throw runtime_error("No tree to convert to list; tree has not been fit yet");
        }
        vector<tuple<int, variant<float, double, int, bool, string>, float>> list;
        return treelist(*tree, list);
    }

};

// FairRandomForestClassifier
class FairRandomForestClassifier {
private:
    int n_jobs;
    int n_bins;
    int max_depth;
    bool bootstrap;
    int random_state;
    int n_estimators;
    float orthogonality;
    int min_samples_leaf;
    int min_samples_split;
    variant<float, string> max_features;
    float sampling_proportion;
    bool hash_values;

    vector<FairDecisionTreeClassifier> trees;
    
    template <class T>
    inline vector<vector<T>> make_batches(const vector<T> iterable) {
        if (n_jobs == -1) {
            n_jobs = thread::hardware_concurrency();
        }
        const auto n_trees = iterable.size();
        if (n_trees < n_jobs) {
            n_jobs = n_trees;
        }
        auto batches = vector<vector<T>>(n_jobs);
        for (int i = 0; i < n_trees; ++i) {
            batches[i % n_jobs].push_back(iterable[i]);
        }
        return batches;
    }

    template <class T>
    inline vector<vector<T>> make_batches(const unordered_set<T> iterable) {
        const vector<T> vi = vector<T>(iterable.begin(), iterable.end());
        return make_batches(vi);
    }

    void fit_batch(const vector<int> batch_states, const vector<vector<variant<float, double, int, bool, string>>>& x, const vector<bool> y, const vector<vector<string>>& s, const vector<vector<string>>& fit_params, mutex & tree_mutex) {
        vector<FairDecisionTreeClassifier> ret;
        for (auto i : batch_states) {
            ret.push_back(
                FairDecisionTreeClassifier(
                    n_bins,
                    max_depth,
                    bootstrap,
                    i,
                    orthogonality,
                    max_features,
                    min_samples_leaf,
                    min_samples_split,
                    sampling_proportion,
                    hash_values
                )
            );
        }
        for (auto& tree : ret) {
            tree.fit(x, y, s, fit_params);
        }
        
        unique_lock<mutex> lock(tree_mutex);
        trees.insert(trees.end(), ret.begin(), ret.end());
    }
    
    void predict_proba_batch(vector<FairDecisionTreeClassifier> batch, vector<vector<variant<float, double, int, bool, string>>> & x, const string mean_type, valarray<float> & y_prob, mutex & y_mutex) {
        valarray<float> result = valarray<float>(0.0, x.size());
        if (mean_type == "prob") {
            for (auto & tree : batch) {
                result += tree.predict_proba(x);
            }
        }
        else if (mean_type == "pred") {
            for (auto & tree : batch) {
                valarray<bool> probtemp = tree.predict(x);
                valarray<float> floattemp = [&]() {
                    valarray<float> ret = valarray<float>(probtemp.size());
                    for (int j = 0; j < probtemp.size(); ++j) ret[j] = (float)probtemp[j];
                    return ret;
                }();
                result += floattemp;
            }
        }
        unique_lock<mutex> lock(y_mutex);
        y_prob += result;
    }


public:
    FairRandomForestClassifier(
        int n_jobs_,
        int n_bins_,
        int max_depth_,
        bool bootstrap_,
        int random_state_,
        int n_estimators_,
        float orthogonality_,
        int min_samples_leaf_,
        int min_samples_split_,
        variant<float, string> max_features_,
        float sampling_proportion_,
        bool hash_values_
    ) :
        n_jobs(n_jobs_),
        n_bins(n_bins_),
        max_depth(max_depth_),
        bootstrap(bootstrap_),
        random_state(random_state_),
        n_estimators(n_estimators_),
        orthogonality(orthogonality_),
        min_samples_leaf(min_samples_leaf_),
        min_samples_split(min_samples_split_),
        max_features(max_features_),
        sampling_proportion(sampling_proportion_),
        hash_values(hash_values_)
    {};

    float pred_th;
    bool is_fit{ false };

    void fit(
        vector<vector<variant<float, double, int, bool, string>>>& x,
        const vector<bool>& y,
        const vector<vector<string>>& s,
        const vector<vector<string>>& fit_params) {


        unordered_set<int> random_states;
        default_random_engine randomeng(random_state);
        while (random_states.size() < n_estimators) {
            random_states.insert(randomeng());
        }

        pred_th = (float)count(y.begin(), y.end(), 1) / (float)y.size();
        trees.reserve(n_estimators);

        if (n_estimators == 1) n_jobs = 1;
        if (n_jobs == 1) {
            for (const auto randomstate : random_states) {
                trees.push_back(
                    FairDecisionTreeClassifier(
                        n_bins,
                        max_depth,
                        bootstrap,
                        randomstate,
                        orthogonality,
                        max_features,
                        min_samples_leaf,
                        min_samples_split,
                        sampling_proportion,
                        hash_values
                    )
                );
            }
            for (auto & tree : trees) {
                tree.fit(x, y, s, fit_params);
            }
        }
        else {
            // We pass the list of random states to the threads, to create the least amount of shared memory between threads.
            // The trees themselves are created inside the threads
            auto batches_states = make_batches(random_states);
            mutex tree_mutex;
            auto thread_pool = vector<thread>();
            for (auto batch_states : batches_states) {
                thread_pool.push_back(thread(&FairRandomForestClassifier::fit_batch, this, batch_states, cref(x), y, cref(s), cref(fit_params), ref(tree_mutex)));
            }
            for (auto& t : thread_pool) {
                t.join();
            }
        }
        is_fit = true;
    }

    valarray<float> predict_proba(vector<vector<variant<float, double, int, bool, string>>>& x, const string mean_type) {
        valarray<float> y_prob = valarray<float>(0.0, x.size());
        if (n_jobs == 1) {
            if (mean_type == "prob") {
                for (auto & tree : trees) {
                    y_prob += tree.predict_proba(x);
                }
                return y_prob / (float)n_estimators;
            }
            else if (mean_type == "pred") {
                for (auto & tree : trees) {
                    valarray<bool> probtemp = tree.predict(x);
                    valarray<float> floattemp = [&]() {
                        valarray<float> ret = valarray<float>(probtemp.size());
                        for (int j = 0; j < probtemp.size(); ++j) ret[j] = (float)probtemp[j];
                        return ret;
                    }();
                    y_prob += floattemp;
                }
                return y_prob / (float)n_estimators;
            }
        }
        else {
            vector<vector<FairDecisionTreeClassifier>> batches_trees = make_batches(trees);
            auto thread_pool = vector<thread>();
            mutex y_mutex;
            for (auto& batch : batches_trees) {
                thread_pool.push_back(thread(&FairRandomForestClassifier::predict_proba_batch, this, ref(batch), ref(x), mean_type, ref(y_prob), ref(y_mutex)));
            }
            for (auto& t : thread_pool) {
                t.join();
            }
            return y_prob / (float)n_estimators;
        }
    }

    vector<vector<float>> predict_proba_wrapper(vector<vector<variant<float, double, int, bool, string>>> x, const string mean_type) {
        auto probs = predict_proba(x, mean_type);
        vector<vector<float>> ret;
        for (int i = 0; i < probs.size(); ++i) {
            ret.push_back({ (float)1 - probs[i], probs[i] });
        }
        return ret;
    }

    valarray<bool> predict(vector<vector<variant<float, double, int, bool, string>>>& x, const string mean_type) {
        return predict_proba(x, mean_type) >= ((0.5 + pred_th) / (float)2);
    }
};


void test() {
    
}


PYBIND11_MODULE(superfastcode, m) {
    py::class_<TreeNode>(m, "TreeNode")
        .def(py::pickle(
            [](TreeNode & t) {
                auto list = vector<tuple<int, variant<float, double, int, bool, string>, float>>();
                return treelist(t, list);
            },
            [](vector<tuple<int, variant<float, double, int, bool, string>, float>> & t) {
                TreeNode tree = unpack_treelist(t);
                return tree;
            }
        ));
    py::class_<FairDecisionTreeClassifier>(m, "FDTC")
        .def(py::init<int, int, bool, int, float, variant<float, string>, int, int, float, bool>(),
            py::arg("n_bins") = 256,
            py::arg("max_depth") = 7,
            py::arg("bootstrap") = false,
            py::arg("random_state") = 42,
            py::arg("orthogonality") = 0.5,
            py::arg("max_features") = 1.0,
            py::arg("min_samples_leaf") = 3,
            py::arg("min_samples_split") = 7,
            py::arg("sampling_proportion") = 1.0,
            py::arg("hash_values") = true
        )
        .def("fit", &FairDecisionTreeClassifier::fit, R"(X -> 2 dimensional list or np.array : numerical and/or categorical
y -> 1 dimensional list or np.array : binary
s -> 2 dimensional list or np.array : categorical)",
            py::arg("X"),
            py::arg("y"),
            py::arg("s") = std::vector<vector<string>>(),
            py::arg("fit_params") = std::vector<vector<string>>())
        .def("predict_proba", &FairDecisionTreeClassifier::predict_proba_wrapper, R"(Retuns the predicted probabilties of input X
X -> 2 dimensional list or np.array : numerical and/or categorical)",
            py::arg("X"))
        .def("predict", &FairDecisionTreeClassifier::predict, R"(Retuns the predicted class label of input X
X -> 2 dimensional list or np.array : numerical and/or categorical)",
            py::arg("X"))
        .def("get_params", &FairDecisionTreeClassifier::get_params,
            py::arg("deep") = false)
        .def("set_params", &FairDecisionTreeClassifier::set_params, 
            py::arg("n_bins") = NULL,
            py::arg("max_depth") = NULL,
            py::arg("bootstrap") = NULL,
            py::arg("max_features") = NULL,
            py::arg("random_state") = NULL,
            py::arg("orthogonality") = NULL,
            py::arg("min_samples_leaf") = NULL,
            py::arg("min_samples_split") = NULL,
            py::arg("sampling_proportion") = NULL,
            py::arg("hash_values") = NULL)
        .def("__str__", &FairDecisionTreeClassifier::__str__)
        .def("__repr__", &FairDecisionTreeClassifier::__repr__)
        .def("to_list", &FairDecisionTreeClassifier::to_list)
        .def(py::pickle(
            [](FairDecisionTreeClassifier& t) {
                return py::make_tuple(t.get_params(false), t.categorical, t.selected_features, *t.tree, t.is_fit);
            },
            [](py::tuple t) {
                if (t.size() != 5) {
                    throw runtime_error("Invalid State!");
                }

                FairDecisionTreeClassifier f = FairDecisionTreeClassifier(
                    t[0]["n_bins"].cast<int>(),
                    t[0]["max_depth"].cast<int>(),
                    t[0]["bootstrap"].cast<bool>(),
                    t[0]["random_state"].cast<int>(),
                    t[0]["orthogonality"].cast<float>(),
                    t[0]["max_features"].cast<variant<float, string>>(),
                    t[0]["min_samples_leaf"].cast<int>(),
                    t[0]["min_samples_split"].cast<int>(),
                    t[0]["sampling_proportion"].cast<float>(),
                    t[0]["hash_values"].cast<bool>()
                );
                f.categorical = t[1].cast<map<int, bool>>();
                f.selected_features = t[2].cast<unordered_set<int>>();
                stringstream s;
                s << t[3].get_type();
                if (s.str() == "<class 'superfastcode.TreeNode'>") {
                    auto a = t[3].cast<TreeNode>();
                    f.tree = make_shared<TreeNode>(a);
                }
                f.is_fit = t[4].cast<bool>();
                return f;
            }
        ))
        .doc() = R"(Fair Decisiont Tree Classifier 
n_bins -> int : 
    feature quantiles from which candidate splits are generated 
min_samples_split -> int : 
    smallest number of samples in a node for a split to be considered
min_samples_leaf -> int :
    smallest number of samples in each leaf after a split for that split to be considered
max_depth -> int :
    max number of allowed splits per tree
sampling_proportion -> float :
    proportion of samples to resample in each tree
max_features -> int :
    number of samples to bootstrap
             -> float :
    proportion of samples to bootstrap
             -> str :
    "auto" / "sqrt" : sqrt of the number of features is used
    "log" / "log2" : log2 of the number of features is used
bootstrap -> bool :
    bootstrap strategy with (True) or without (False) replacement
random_state -> int :
    seed for all random processes
orthogonality -> float :
    strength of fairness constraint in which :
    0 is no fairness constraint(i.e., 'traditional' classifier)
    [0, 1]
hash_values -> bool :
    strategy for handling string values with (True) or without (False) hashing)";
        py::class_<FairRandomForestClassifier>(m, "FRFC")
            .def(py::init<int, int, int, bool, int, int, float, int, int, variant<float, string>, float, bool>(),
                py::arg("n_jobs") = -1,
                py::arg("n_bins") = 256,
                py::arg("max_depth") = 7,
                py::arg("bootstrap") = true,
                py::arg("random_state") = 42,
                py::arg("n_estimators") = 500,
                py::arg("orthogonality") = 0.5,
                py::arg("min_samples_leaf") = 3,
                py::arg("min_samples_split") = 7,
                py::arg("max_features") = "auto",
                py::arg("sampling_proportion") = 1.0,
                py::arg("hash_values") = true)
            .def("fit", &FairRandomForestClassifier::fit, R"(X -> 2 dimensional list or np.array : numerical and/or categorical
y -> 1 dimensional list or np.array : binary
s -> 2 dimensional list or np.array : categorical)",
                py::arg("X"),
                py::arg("y"),
                py::arg("s") = std::vector<vector<string>>(),
                py::arg("fit_params") = std::vector<vector<string>>())
            .def("predict_proba", &FairRandomForestClassifier::predict_proba_wrapper, R"(Retuns the predicted probabilties of input X
X -> 2 dimensional list or np.array : numerical and/or categorical
mean_type -> str
    Method to compute the probailities across all trees
    {"prob", "pred"}
    "prob" computes the mean of all tree probabilities (the probability of Y=1 of each terminal node)
    "pred" computes the mean of all tree predicitons {0, 1})",
                py::arg("X"),
                py::arg("mean_type") = "prob")
            .def("predict", &FairRandomForestClassifier::predict, R"(Retuns the predicted class label of input X
X -> 2 dimensional list or np.array : numerical and/or categorical
mean_type -> str
    Method to compute the probailities across all trees
    {"prob", "pred"}
    "prob" computes the mean of all tree probabilities (the probability of Y=1 of each terminal node)
    "pred" computes the mean of all tree predicitons {0, 1})", 
                py::arg("X"),
                py::arg("mean_type") = "prob")
            .doc() = R"(Fair Random Forest Classifier 
n_estimators -> int: 
    number of FairDecisionTreeClassifier objects 
n_bins -> int : 
    feature quantiles from which candidate splits are generated 
min_samples_split -> int : 
    smallest number of samples in a node for a split to be considered
min_samples_leaf -> int :
    smallest number of samples in each leaf after a split for that split to be considered
max_depth -> int :
    max number of allowed splits per tree
sampling_proportion -> float :
    proportion of samples to resample in each tree
max_features -> int :
    number of samples to bootstrap
             -> float :
    proportion of samples to bootstrap
             -> str :
    "auto" / "sqrt" : sqrt of the number of features is used
    "log" / "log2" : log2 of the number of features is used
bootstrap -> bool :
    bootstrap strategy with (True) or without (False) replacement
random_state -> int :
    seed for all random processes
orthogonality -> float :
    strength of fairness constraint in which :
    0 is no fairness constraint(i.e., 'traditional' classifier)
    [0, 1]
n_jobs -> int :
    CPU usage; -1 for all threads
hash_values -> bool :
    strategy for handling string values with (True) or without (False) hashing)";
        m.def("test", &test);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}   