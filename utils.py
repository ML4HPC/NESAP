import torch
import numpy as np;
from torch.autograd import Variable
import torch.distributed as dist


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 2, amp_size = 1):
        
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        #fin = open(file_name);
        #rawdat_tmp = np.loadtxt(fin,delimiter=',');
        rawdat_tmp = np.load(file_name) # .npy file
        self.rawdat = rawdat_tmp

        for i in range(1,amp_size):
            self.rawdat = np.concatenate((self.rawdat, rawdat_tmp))

        #self.rawdat = np.loadtxt(fin,delimiter=',');
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m1, self.m2, self.m3 = self.dat.shape; # 3D case

        #print('n : ', self.n)
        #print('m : ', self.m)

        self.normalize = 2
        self.scale = np.ones((self.m1, self.m2, self.m3));
        self._normalized(normalize);
        self._split(int(train * self.n), int((train+valid) * self.n), self.n);
        
        self.scale = torch.from_numpy(self.scale).float();
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m1, self.m2, self.m3);
            
        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));
    
    def _normalized(self, normalize):
        #normalized by the maximum value of entire matrix.
       
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat);
            
        # 3D normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m1):
                for j in range(self.m2):
                    for k in range(self.m3):
                        self.scale[i, j, k] = np.max(np.abs(self.rawdat[:, i, j, k]));
                        self.dat[:,i, j, k] = self.rawdat[:,i, j, k] / np.max(np.abs(self.rawdat[:,i, j, k]));
            
        
    def _split(self, train, valid, test):
        train_set = range(self.P+self.h-1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.n);
        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);
        self.test = self._batchify(test_set, self.h);
        
        
    def _batchify(self, idx_set, horizon):
        
        n = len(idx_set);
        X = torch.zeros((n,self.P,self.m1, self.m2, self.m3));
        Y = torch.zeros((n,self.m1, self.m2, self.m3));
        
        for i in range(n):
            end = idx_set[i] - self.h + 1;
            start = end - self.P;
            X[i,:,:, :, :] = torch.from_numpy(self.dat[start:end, :]);
            Y[i,:, :, :] = torch.from_numpy(self.dat[idx_set[i], :]);

        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        
        rank = dist.get_rank()
        wsize = dist.get_world_size()

        start_idx = rank*batch_size;
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();  
            yield Variable(X), Variable(Y);
            start_idx += batch_size*wsize;
