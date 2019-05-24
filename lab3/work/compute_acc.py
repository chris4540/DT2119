"""
For part 5 evaluation

Compute the phoneme acc instead of frame-by-frame at the phoneme level

For windows:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-levenshtein
"""
import numpy as np
import Levenshtein
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.viridis):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        # xticks=np.arange(cm.shape[1]),
        # yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    fig.tight_layout()
    return ax

def list_idx_to_str(list_):
    ret = ''.join(str(x) for x in list_)
    return ret

def levenshtein_dist(s, t):
    s_string = list_idx_to_str(s)
    t_string = list_idx_to_str(t)
    ret = Levenshtein.distance(s_string, t_string)
    return ret

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
    # dynamic = True
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
    ax = plot_confusion_matrix(test_y, pred_y)
    ax.set_title("Confusion matrix at state level")
    fig = ax.get_figure()
    fig.savefig("{}_state_cm.png".format(tag), bbox_inches='tight')
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

    # =============================================================
    # build mapping from state to phone (index)
    with open("data/state_list.json", 'r') as f:
        state_list = json.load(f)['state_list']


    map_state_to_phone_idx = dict()
    phone_idx = dict()
    for i, s in enumerate(state_list):
        phone = s.split("_")[0]
        if phone not in phone_idx:
            phone_idx[phone] = len(phone_idx)
        # ====================================
        map_state_to_phone_idx[i] = phone_idx[phone]
    # ================================================================
    # frame-by-frame at the phoneme level
    v_map = np.vectorize(map_state_to_phone_idx.get)
    pred_y_phone = v_map(pred_y)
    test_y_phone = v_map(test_y)
    test_acc = (pred_y_phone == test_y_phone).mean()
    print("[{}] Frame-by-frame acc. at the phoneme level = ".format(tag), test_acc)
    # =================================================================
    ax = plot_confusion_matrix(test_y_phone, pred_y_phone)
    ax.set_title("Confusion matrix at phoneme level")
    fig = ax.get_figure()
    fig.savefig("{}_phone_cm.png".format(tag), bbox_inches='tight')
    # ==============================================================
    # compute the avg. edit distance
    cnt = 0
    edit_dists = list()
    for d in tqdm(test_data):
        targets = d['targets']
        targets = v_map(targets)
        end_idx = cnt + len(targets)
        y_pred_this = pred_y[cnt:end_idx]
        # mapping
        y_pred_this = v_map(y_pred_this)
        target_merged = merge_conseq_ident_seq(targets)
        y_merged = merge_conseq_ident_seq(y_pred_this)
        dist = levenshtein_dist(target_merged, y_merged)
        dist /= max(len(target_merged), len(y_merged))
        edit_dists.append(dist)
        #
        cnt = end_idx
    print("[{}] Frame-by-frame edit dist avg. at the phoneme level = ".format(tag), np.mean(edit_dists))
