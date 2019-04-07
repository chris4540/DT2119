from sklearn.mixture import GaussianMixture
from feature_corr import concat_all_features
from lab1_proto import mfcc

def get_test_data(data, digit):
    """
    Args:
        digit: the string of the target digit
    """
    ret = None
    for d in data:
        if d['digit'] == digit:
            features = mfcc(d['samples'], samplingrate=d['samplingrate'])

            if ret is None:
                ret = features
            else:
                ret = np.concatenate((ret, features), axis=0)
    return ret

if __name__ == "__main__":
    data = np.load('data/lab1_data.npz')['data']
    all_features = concat_all_features(data, feature="mfcc")

    test_data = get_test_data(data, digit='7')

    gmm = GaussianMixture(32, covariance_type='diag', verbose=1)
    # train
    gmm.fit(all_features)

    # prediction
