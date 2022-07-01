from __future__ import print_function
import torch
import torch.utils.data as data
import random
import pickle
import gc
import sys
sys.path.append('../architectures/')


# ------------------------ #
# --- MM preprocessing --- #
# ------------------------ #
def preprocess_triplet_data_seed(filename,seed):
    with open('datasets/'+filename, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list = pickle.load(f)
        else:
            data_list = pickle.load(f, encoding='latin1')

    random.seed(int(seed))
    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    print('Creating train split')
    train_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    train_data))
    with open('datasets/train_'+filename[:-4] + "_seed_" + str(seed) + ".pkl", 'wb') as f:
        pickle.dump(train_data1, f)
    print('train split created')
    del train_data1
    gc.collect()
    
    print('Creating test split')
    test_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    test_data))
    with open('datasets/test_'+filename[:-4] + "_seed_" + str(seed) + ".pkl", 'wb') as f:
        pickle.dump(test_data1, f)
    print('test split created')

# ----------------------- #
# --- Custom Datasets --- #
# ----------------------- #
class TripletTensorDataset(data.Dataset):
    def __init__(self, dataset_name, split):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split

        try:
            if split == 'test':
                with open('datasets/test_'+self.dataset_name+'.pkl', 'rb') as f:
                    self.data = pickle.load(f)
            else:
                with open('datasets/train_'+self.dataset_name+'.pkl', 'rb') as f:
                    self.data = pickle.load(f)

        except:
            raise ValueError('Not recognized dataset {0}'.format(self.dataset_name))

    def __getitem__(self, index):
        img1, img2, action = self.data[index]
        return img1, img2, action

    def __len__(self):
        return len(self.data)
    

class TripletTensorDatasetClassesAct(data.Dataset):
    def __init__(self, dataset_name):

        self.dataset_name =  dataset_name
        self.name = self.dataset_name + '_'

        with open("datasets/"+self.dataset_name+'.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        img1, img2, action, a_lambda, class1, class2 = self.data[index]
        return img1, img2, action, a_lambda, class1, class2

    def __len__(self):
        return len(self.data)


class APNDataset(data.Dataset):
    def __init__(self, task_name, dataset_name, split, random_seed, dtype,
                 img_size):
        self.task_name = task_name
        self.dataset_name =  dataset_name
        self.name = dataset_name + '_' + split
        self.split = split.lower()
        self.random_seed = random_seed
        self.dtype = dtype
        self.img_size = img_size

        path = 'datasets/action_data/{0}/{1}_{2}_seed{3}.pkl'.format(
                self.dataset_name, self.dtype, self.split, self.random_seed)

        with open(path, 'rb') as f:
            pickle_data = pickle.load(f)
            self.data = pickle_data['data']
            self.min, self.max = pickle_data['min'], pickle_data['max']


    def __getitem__(self, index):
        img1, img2, coords = self.data[index]
        return img1, img2, coords

    def __len__(self):
        return len(self.data)
