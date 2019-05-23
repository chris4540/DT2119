"""
Data normalization for 4.6
"""
import numpy as np
from tqdm import tqdm
def get_mean_std(data, feature):
   # Obtain all data
   all_X = list()
   for d in tqdm(data):
      all_X.append(d[feature])
   # concat
   X_full = np.concatenate(all_X, axis=0)
   # cal mean and std
   mean_X = np.mean(X_full, axis=0)
   std_X = np.std(X_full,  axis=0)
   return mean_X, std_X


if __name__ == "__main__":
   feature_name = 'lmfcc'
   # load data
   data = np.load('data/train_val_data.npz')
   valdata = data['validation']
   traindata = data['train']
   testdata = np.load('data/testdata.npz')['testdata']

   # calculate the mean and std from train data
   mean, std = get_mean_std(traindata, feature_name)
   print("Complete calculating mean and std")
   datasets = {
      'train': traindata,
      'test': testdata,
      'val': valdata,
   }

   for k, v in datasets.items():
      data = list()
      for d in tqdm(v):
         new_data = dict()
         new_data['filename'] = d['filename']
         new_data['targets'] = d['targets']
         new_data['lmfcc'] = (d['lmfcc'] - mean) / std
         data.append(new_data)

      print("Complete normlaizating ", k)
      # save it
      np.savez('data/nondynamic/{}_{}.npz'.format(feature_name, k), data=data)




