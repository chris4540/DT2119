"""
Part 5 Phoneme Recognition with Deep Neural Networks

Basic implmentation
"""
import numpy as np
import json
import keras
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

class Config:
    batch_size = 300
    epochs = 5
    activation = 'relu'
    # activation = 'tanh'
    # activation = 'sigmoid'

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation=Config.activation))
    model.add(Dense(256, activation=Config.activation))
    model.add(Dense(256, activation=Config.activation))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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


if __name__ == "__main__":
    feature = "lmfcc"
    dynamic = False
    # =========================================================
    if dynamic:
        dyn_tag = "dyn"
    else:
        dyn_tag = "nondyn"

    tag = "_".join([feature, dyn_tag, Config.activation])
    print("Exp: ", tag)
    # =========================================================
    # load statelist
    with open('data/state_list.json', 'r') as f:
        state_list = json.load(f)['state_list']
    output_dim = len(state_list)
    path_template = "data/%s/%s_{dtype}.npz" % (dyn_tag, feature)
    # Load data
    train_data = np.load(path_template.format(dtype="train"))['data']
    val_data = np.load(path_template.format(dtype="val"))['data']
    test_data = np.load(path_template.format(dtype="test"))['data']

    # construct input data and labels into a large array
    train_x, train_y = parse_data(train_data, feature)
    val_x, val_y = parse_data(val_data, feature)
    test_x, test_y = parse_data(test_data, feature)
    # one-hot encoding
    train_y = np_utils.to_categorical(train_y, output_dim)
    val_y = np_utils.to_categorical(val_y, output_dim)
    test_y = np_utils.to_categorical(test_y, output_dim)

    print("End constructing data")
    # ============================================
    input_dim = train_x.shape[1]
    print("===============================")
    print("Input dim = ", input_dim)
    print("===============================")
    model = build_model(input_dim, output_dim)
    train_log = model.fit(
        train_x, train_y,
        batch_size=Config.batch_size, epochs=Config.epochs,
        validation_data=(val_x, val_y))


    np.savez(tag + "_trainlog.npz", train_log=train_log)
    test_class_prob = model.predict(test_x)
    np.savez(tag + "_test_class_prob.npz", train_log=test_class_prob)

    score, acc = model.evaluate(val_x, val_y, batch_size=Config.batch_size)
    print("Final validation score and acc.")
    print(score, acc)

    score, acc = model.evaluate(test_x, test_y, batch_size=Config.batch_size)
    print("Final test score and acc.")
    print(score, acc)
