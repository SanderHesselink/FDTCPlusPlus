import tools

def test():
    X, y, s = tools.make_set(5, 1000, 10, [0.2, 0.5])
    tools.save(X, y, s, "test")

def large_num():
    X, y, s = tools.make_set(5, 10000, 10, [0.2, 0.5])
    tools.save(X, y, s, "large_num")

def large_string():
    X, y, s = tools.make_set(5, 10000, 10, [0.2, 0.5])
    for i in range(len(X)):
        for j in range(len(X[i])):
            for k in range(len(X[i][j])):
                X[i][j][k] = str(X[i][j][k])


    tools.save(X, y, s, "large_string")

def large_num_complex():
    X, y, s = tools.make_set(5, 10000, 10, [0.2, 0.5], info_ratio=1, redun_ratio=0, sep=0.8)
    tools.save(X, y, s, "large_num_complex")

def large_num_unbalanced():
    X, y, s = tools.make_set(5, 10000, 10, [0.2, 0.5], balance=[0.1])
    tools.save(X, y, s, "large_num_unbalanced")

def large_one_d():
    X, y, s = tools.make_set(1, 10000, 10, [0.3])
    tools.save(X, y, s, "large_one_d")

def big_one_d():
    X, y, s = tools.make_set(1, 100000, 10, [0.3])
    tools.save(X, y, s, "big_one_d")

def huge_one_d():
    X, y, s = tools.make_set(1, 1000000, 10, [0.3])
    tools.save(X, y, s, "huge_one_d")

def complex_variable(n, m, s):
    X, y, s = tools.make_set(5, n, m, [s], info_ratio=1, redun_ratio=0, sep=0.8, flip=0.15)
    return X, y, s