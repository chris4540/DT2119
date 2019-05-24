"""
For part 5 evaluation

Compute the phoneme acc instead of frame-by-frame at the phoneme level

For windows:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-levenshtein
"""
import numpy as np
import Levenshtein
from tqdm import tqdm

def list_idx_to_str(list_):
    ret = ''.join(str(x) for x in list_)
    return ret

def levenshtein_dist(s, t):
    s_string = list_idx_to_str(s)
    t_string = list_idx_to_str(t)
    ret = Levenshtein.distance(s_string, t_string)
    return ret

# def levenshtein_dist(s, t):
#     if not s: # consider the empty list
#         return len(t)
#     if not t:  # consider the empty list
#         return len(s)

#     # compare the last components
#     if s[-1] == t[-1]:
#         cost = 0
#     else:
#         cost = 1

#     # pop out the last one, compare recursively by dynamic programming
#     res = min([levenshtein_dist(s[:-1], t)+1,
#                levenshtein_dist(s, t[:-1])+1,
#                levenshtein_dist(s[:-1], t[:-1]) + cost])
#     return res

def parse_data(data, feature):
    """
    Construct input data and labels into a large array
    """
    X_list = []
    y_list = []
    for d in data:
        X_list.append(d[feature])
        y_list.append(d['targets'])
    x_mat = np.concatenate(X_list, axis=0).astype('float32')
    y_mat = np.concatenate(y_list, axis=0)
    return x_mat, y_mat

def merge_conseq_ident_seq(seq):
    """
    Merge consequent identical elements
    E.g.
        [0, 0, 1, 2, 2, 3, 1] => [0, 1, 2, 3, 1]
    """
    ret = list()
    for i in seq:
        if not ret:
            ret.append(i)
        elif ret[-1] != i:
            ret.append(i)
    return ret


if __name__ == "__main__":
    feature = "lmfcc"
    # feature = "mspec"
    dynamic = False
    # =========================================================
    if dynamic:
        dyn_tag = "dyn"
    else:
        dyn_tag = "nondyn"
    tag = "_".join([feature, dyn_tag, 'relu'])
    path_template = "data/%s/%s_{dtype}.npz" % (dyn_tag, feature)
    file_ = "results/{}_test_class_prob.npz".format(tag)
    test_class_prob = np.load(file_)['train_log']

    test_data = np.load(path_template.format(dtype="test"))['data']
    _, test_y = parse_data(test_data, feature)
    # ==============================================================
    # calculate the predicted class
    pred_y = np.argmax(test_class_prob, axis=1)
    test_acc = (pred_y == test_y).mean()
    print("[{}] Frame-by-frame acc. = ".format(tag), test_acc)
    # ==============================================================
    # compute the avg. edit distance
    cnt = 0
    edit_dists = list()
    for d in tqdm(test_data):
        targets = d['targets']
        end_idx = cnt + len(targets)
        y_pred_this = pred_y[cnt:end_idx]
        target_merged = merge_conseq_ident_seq(targets)
        y_merged = merge_conseq_ident_seq(y_pred_this)
        dist = levenshtein_dist(target_merged, y_merged)
        dist /= max(len(target_merged), len(y_merged))
        edit_dists.append(dist)
        #
        cnt = end_idx

    print("[{}] Frame-by-frame edit dist avg. = ".format(tag), np.mean(edit_dists))





    # # 1. merging
    # pred_y_merged = merge_conseq_ident_seq(pred_y)
    # test_y_merged = merge_conseq_ident_seq(test_y)
    # edit_dist = levenshtein_dist(pred_y_merged, test_y_merged)
    # print(edit_dist)