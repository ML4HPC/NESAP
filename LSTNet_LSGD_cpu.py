import argparse
import math
import time
import sys
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from models import LSTNet

import numpy as np;
import importlib


from utils import *;
import Optim



def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    #print('===== evaluate() =====')
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        #if not predict:
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));
        
        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).data[0]
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data[0]
        n_samples += (output.size(0) * data.m);
    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae
    
    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0);
    try:
        correlation = correlation /(sigma_p * sigma_g);
    except (ZeroDivisionError, FloatingPointError, UnboundLocalError):
        print('Zero Division Error')
        print('sigma_p * sigma_g = \n', sigma_p * sigma_g)
    correlation = (correlation[index]).mean();
    return rse, rae, correlation;


def avg_grad(model, iscuda):
    rank = dist.get_rank()
    size = float(dist.get_world_size())
    #if iscuda:
        #start_time = time.time()
        #model.cpu()
        #end_time = time.time()
        #print('Copy from GPU to CPU : ', end_time - start_time)

    #dist.barrier();
    for param in model.parameters():
        #print('RANK[',rank,']: param.nelement() : ', param.nelement(), ', param.grad.data type : ', type(param.grad.data), ', param.grad.data size : ' , param.grad.data.size())
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        #print('RANK[',rank,']: avg_grad()? - 2')
        #param.grad.data /= size

    #if iscuda:
        #start_time = time.time()
        #model.cuda()
        #end_time = time.time()
        #print('Copy from CPU to GPU : ', end_time - start_time)







def reduce_loss(total_loss, n_samples):
    reduction = torch.FloatTensor([[total_loss],[n_samples]])
    dist.all_reduce(reduction, op=dist.ReduceOp.SUM)
    return float(reduction[0] / reduction[1])


def train(comm_rank, node_handle, node_idx, comm_handle, data, X, Y, model, criterion, optim, batch_size, iscuda, dataset_size, iter_size, remainder):
    #print('Here ? - 1')
    rank = dist.get_rank()
    wsize = dist.get_world_size()
    model.train();
    total_loss = 0;
    n_samples = 0;

    length = dataset_size





    if comm_rank:

        #start = time.time()
        for i in range(iter_size):


            for param in model.parameters():
                param.data.zero_()
                dist.reduce(param.data, local_root, op=dist.ReduceOp.SUM, group=node_handle[node_idx])
                param.data /= float(train_proc_num)

            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=comm_handle)


            for param in model.parameters():
                dist.broadcast(param.data, local_root, group=node_handle[node_idx])


    else:


        index = torch.randperm(length)
        start_idx = 0;


        for i in range(iter_size):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X_ = X[excerpt];
            Y_ = Y[excerpt];
            X_ = Variable(X_)
            Y_ = Variable(Y_);
            start_idx += batch_size;


            if i > 0:
                for param in model.parameters():
                    dist.broadcast(param.grad.data, local_root, group=node_handle[node_idx])

                optim.step()
                total_loss += loss.data.item();
                n_samples += (output.size(0) * data.m);



            model.zero_grad();
            output = model(X_);
            scale = data.scale.expand(output.size(0), data.m)
            loss = criterion(output * scale, Y_ * scale);
            loss.backward();

            #avg_grad(model, iscuda);
            for param in model.parameters():
                dist.reduce(param.grad.data, local_root, op=dist.ReduceOp.SUM, group=node_handle[node_idx])



        for param in model.parameters():
            dist.broadcast(param.grad.data, local_root, group=node_handle[node_idx])

            optim.step()
            total_loss += loss.data.item();
            n_samples += (output.size(0) * data.m);



    #print('Here ? - 4')
    return reduce_loss(total_loss, n_samples)
   







dist.init_process_group('mpi')

rank = dist.get_rank()
wsize = dist.get_world_size()
world_size = wsize





 
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data file')
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24 * 7, help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--iter_size', type=int, default=8, metavar='N', help='iteration size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321, help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt', help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--skip', type=float, default=24)
#parser.add_argument('--skip', type=int, default=6)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
parser.add_argument('--data_amp_size', type=int, default=1)
args = parser.parse_args()

args.cuda = args.gpu is not None
if args.cuda:
    #torch.cuda.set_device(args.gpu)
    if local_rank < args.gpu:
        torch.cuda.set_device(local_rank)
    else:
        args.cuda=False



# Set the random seed manually for reproducibility.
#torch.manual_seed(args.seed)
#if torch.cuda.is_available():
#    if not args.cuda:
#        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#    else:
#        torch.cuda.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed*rank)
else:
    torch.manual_seed(args.seed*rank)

Data = Data_utility(args.data, 0.9, 0.05, args.cuda, args.horizon, args.window, args.normalize, args.data_amp_size);
#print(Data.rse);

#print('Data_utility creation completed!!!')

model = eval(args.model).Model(args, Data);

if args.cuda:
    model.cuda()
    
nParams = sum([p.nelement() for p in model.parameters()])
if rank==0: print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();
    
   
 
best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

batch_size = args.batch_size


remainder = 0


#local_size = (int)(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
#local_rank = (int)(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
local_rank = (int)(os.environ['SLURM_LOCALID'])
local_size = (int)(os.environ['SLURM_NTASKS_PER_NODE'])



dataset_size = len(Data.train[0])
#dataset_size = dataset_size*8
iter_size = (dataset_size*8)//(batch_size*wsize)





node_num = world_size // local_size
train_proc_num = world_size - node_num
node_idx = rank // local_size
local_root = node_idx * local_size






start_idx = 0
end_idx = local_size
node_handle = list()
comm_handle_idx = list()
for i in range(node_num):
    node_handle.append(dist.new_group(range(start_idx, end_idx)))
    comm_handle_idx.append(start_idx)
    start_idx += local_size
    end_idx += local_size

comm_handle = dist.new_group(comm_handle_idx)








#if rank==0: print('GPU Devices # : ', torch.cuda.device_count())
if rank==0: print('comm_handle : ', comm_handle_idx)
if rank==0: print('pytorch version : ', torch.__version__)
if rank==0: print('m : ', Data.m)
if rank==0: print('n : ', Data.n)
if rank==0: print('Training data set length : ', dataset_size)
if rank==0: print('Training iteration size : ', iter_size)
if rank==0: print('Training dataset size per epoch : ', iter_size*batch_size*wsize)
if rank==0: print('WORLD SIZE:', world_size)
if rank==0: print('The number of nodes : ', node_num)
if rank==0: print('The number of ranks in a node : ', local_size)
if rank==0: print('The number of processes doing training : ', train_proc_num)





comm_rank = False
if rank % local_size == 0:
    comm_rank = True




dist.barrier();
if rank==0: print('========================================================')
sys.stdout.flush()
dist.barrier();

print('[',rank,'] : LOCAL_RANK : ', local_rank, ', Node : ', node_idx, ', Worker : ', not comm_rank, ', COMM_RANK : ', comm_rank)
sys.stdout.flush()

dist.barrier();
if rank==0: print('========================================================')
sys.stdout.flush()
dist.barrier();

sys.stdout.flush()


if comm_rank:
    for param in model.parameters():
        param.data.zero_()

for param in model.parameters():
    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    if comm_rank:
        param.data.zero_()
    else:
        param.data /= train_proc_num




dist.barrier();
# At any point you can hit Ctrl + C to break out of training early.
try:
    np.seterr(divide='raise', invalid='raise')
    if rank==0: print('===== Begin Training =====');

    total_training_time = time.time()
    for epoch in range(1, args.epochs+1):
        #print('===== Start of train() =====')
        epoch_start_time = time.time()
        train_loss = train(comm_rank, node_handle, node_idx, comm_handle, Data, Data.train[0], Data.train[1], model, criterion, optim, batch_size, args.cuda, dataset_size, iter_size, remainder)
        epoch_end_time = time.time()
        #print('===== End of train() =====')
        #val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size);
        if rank == 0: print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f}'.format(epoch, (epoch_end_time - epoch_start_time), train_loss))
        sys.stdout.flush()
        #if rank == 0: print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (epoch_end_time - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        #if val_loss < best_val:
        #    with open(args.save, 'wb') as f:
        #        torch.save(model, f)
        #    best_val = val_loss
        #if epoch % 5 == 0:
        #    test_acc, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size);
        #    print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

    if rank == 0: print('Total training time : ', time.time() - total_training_time)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

## Load the best saved model.
#with open(args.save, 'rb') as f:
#    model = torch.load(f)
#test_acc, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size);
#print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))



