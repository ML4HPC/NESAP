import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, data):
        super(Model, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.P = args.window; 
        self.m1, self.m2, self.m3 = data.m1, data.m2, data.m3 # 3 dimensions of the 3D data
        self.hidR = args.hidRNN; 
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip; 
        self.Ck = args.CNN_kernel; 
        self.skip = args.skip; 
        self.pt = (self.P - self.Ck)/self.skip if self.skip > 0 else None
        self.hw = args.highway_window 
        # self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        
        # 3D case
        self.conv1 = nn.Conv3d(self.P, self.P * self.hidC * self.Ck, kernel_size = (self.m1, self.m2, self.m3));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = 0.2);
        
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m1*self.m2*self.m3);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m1*self.m2*self.m3);
            
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
            
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = torch.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, self.P, self.m1, self.m2, self.m3);
        c = F.relu(self.conv1(c));
        c = torch.squeeze(c.view(-1, self.hidC, self.Ck, self.P, 1));
        # print(c.size(), self.hidC, self.Ck, self.P)
        
        res = torch.zeros(c.size(0), self.hidC, self.P-self.Ck+1)
        for batch_index in range(c.size(0)):
            for channel in range(self.hidC):
                for begin in range(self.Ck):
                    res[batch_index, channel, :] += c[batch_index, channel, begin, begin:begin+self.P-self.Ck+1] 
        c = res
        c = self.dropout(c);
        # print(c.shape) # [10, 3, 97]
        # c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0));

        #skip-rnn
        if (self.skip > 0):
            #s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = c[:,:, -int(self.pt) * self.skip:].contiguous();
            # print(s.size(), batch_size, self.hidC, int(self.pt), self.skip)
            s = s.view(batch_size, self.hidC, int(self.pt), self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(int(self.pt), batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        #print(res.shape)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :, :, :];
            z = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            #print(z.shape)
            z = z.view(-1,self.m1*self.m2*self.m3);
            res = res + z;
            
        if (self.output):
            res = self.output(res);
            
        res = res.view(-1, self.m1, self.m2, self.m3)
        #print(res.shape) 
        return res;