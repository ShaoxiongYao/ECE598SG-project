import torch, os
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
# actions (sustitute the following code with your directory of example_dynamics_data)
"""
for d in os.listdir("/home/kaiyan3/ECE598SG-project/data/example_dynamics_data/kitchen"):
    if d.find(".pt") != -1:
        print(d)
        data = torch.load("/home/kaiyan3/ECE598SG-project/data/example_dynamics_data/kitchen/"+d)
        print(data.shape) 
"""



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="0 is direct, 1 is LSTM", default=0,type=int)
    parser.add_argument("--name", help="name of environment", default="kitchen", type=str)
<<<<<<< HEAD
    parser.add_argument("--path", help="path of data", default= "data/example_dynamics_data/")
=======
>>>>>>> ea4937ae0ebd9e3cf4dc1493f5f1441cbc175325
    return parser.parse_args()

# one approach: given s_t and skill_z, predict s_{t+N}

BS = 128 # batch size

class Trajdataset(Dataset):
    def __init__(self, feature, label):
        assert feature.shape[0] == label.shape[0], "Feature and label are not aligned!"
        self.n, self.dim_feature, self.dim_label = feature.shape[0], feature.shape[1], label.shape[1]
        self.feature = feature
        self.label = label 
        
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"feature": self.feature[idx], "label": self.label[idx]}

def get_traj_dataset(feature, label):
    # return two dataloaders.
    
    n = feature.shape[0]
    idx = torch.randperm(n)
    n_train, n_valid = int(n * 0.8), n - int(n * 0.8)
    train_dataset, test_dataset = Trajdataset(feature[:n_train], label[:n_train]), Trajdataset(feature[n_train:], label[n_train:])
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=n_valid, shuffle=False)
    return train_loader, test_loader
    

class PlainNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.middle_dim = 256
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.middle_dim),
            nn.LeakyReLU(),
            nn.Linear(self.middle_dim, self.middle_dim),
            nn.LeakyReLU(),
            nn.Linear(self.middle_dim, self.middle_dim),
            nn.LeakyReLU(),
            nn.Linear(self.middle_dim, self.output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
<<<<<<< HEAD
        
    def predict(self, x):
        return self.forward(x)
=======
>>>>>>> ea4937ae0ebd9e3cf4dc1493f5f1441cbc175325

L = 10

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.middle_dim = 256
        self.nn1 = nn.Linear(self.input_dim, self.middle_dim // 4)
        self.nn2 = nn.Linear(self.middle_dim // 4, self.middle_dim // 4) # 64-dimensional input
        self.lstm = nn.LSTM(self.middle_dim // 4, self.middle_dim // 4, batch_first=True, num_layers=1) # 256 input, 256 hidden,            
        # input is of shape (batch_size, 10, 64)
        # output is of shape (batch_size, 10, 64)
        # https://discuss.pytorch.org/t/how-to-create-a-lstm-with-one-to-many/108659/2
        """
        Information in an LSTM, represented by the state vectors, “vanishes” across the time dimension. If you feed the input features only once, say at the first time step, it’s likely, that they won’t be fully propagated to the later time steps and hence the model would have a bias across the time dimension. Hence, you might want to use the same feature vector as input for every time step.
        """
        self.nn3 = nn.Linear(self.middle_dim // 4, self.output_dim)
         

    def forward(self, x):
        x = F.leaky_relu(self.nn2(F.leaky_relu(self.nn1(x))))
        # x is of shape (batch_size, 64)
        x = self.lstm(x.unsqueeze(1).repeat(1, L, 1))[0] # length is L = 10
        # x is of shape (batch_size, 10, 64)  
        x = self.nn3(x)
        # x is of shape (batch_size, 10, output_dim)
        return x
<<<<<<< HEAD
    
    def predict(self, x):
        return 
=======
>>>>>>> ea4937ae0ebd9e3cf4dc1493f5f1441cbc175325

if __name__ == "__main__":
    
    args = get_args()
<<<<<<< HEAD
    path = args.path +args.name+"/" # change this to your directory! 
=======
    path = "/home/kaiyan3/ECE598SG-project/data/example_dynamics_data/"+args.name+"/" # change this to your directory! 
>>>>>>> ea4937ae0ebd9e3cf4dc1493f5f1441cbc175325
    
    device = torch.device("cuda:0")
    
    states = torch.load(path+"states0.pt", map_location=device)
    skill_z = torch.load(path+"skill_z0.pt", map_location=device)
    # print(torch.norm(states[0, 0, :]))
    if args.mode == 0:
        feature = torch.cat((states[:, 0, :], skill_z), dim=1) # 12325 * 70
        label = states[:, -1, :] # 12325 * 60
        net = PlainNet(feature.shape[1], label.shape[1]).to(device)
    else:   # alternative: given s_t and skill_z, use a LSTM to predict s_{t+1} to s_{t+N}
        feature = torch.cat((states[:, 0, :], skill_z), dim=1)
        label = states[:, 1:, :] # 12325 * 10 * 60
        net = LSTM(feature.shape[1], label.shape[-1]).to(device)
    
    train_loader, test_loader = get_traj_dataset(feature, label)
    optimizer = torch.optim.Adam(net.parameters())
    N = 150 # for kitchen
    valid_losses = []
    for epoch in range(N):
        net.train() 
        avg_loss = 0
        for batch_idx, samples in enumerate(train_loader, 0):
            feature, label = samples["feature"], samples["label"]
            optimizer.zero_grad()
            outputs = net(feature)
            # print("outputs:", (outputs[0] - label[0]))
            loss = nn.MSELoss()(outputs, label)
            # print("epoch", epoch, "training loss:", loss)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        # print("epoch", epoch, "training loss:", avg_loss / len(train_loader))
        net.eval()
        for batch_idx, samples in enumerate(test_loader, 0):
            feature, label = samples["feature"], samples["label"]
            outputs = net(feature)
            loss = nn.MSELoss()(outputs, label)
            valid_losses.append(loss)
            if epoch == N - 1: print(outputs[0], label[0])
            print("test loss:", loss)
<<<<<<< HEAD
        if len(valid_losses) > 3:  # early stopping
            if valid_losses[-1] > valid_losses[-2] and valid_losses[-2] > valid_losses[-3]:
                mode = "MLP" if args.mode == 0 else "LSTM"
                torch.save(net, args.path+args.name+"_"+mode+".pth")
=======
        if len(valid_losses) > 3:
            if valid_losses[-1] > valid_losses[-2] and valid_losses[-2] > valid_losses[-3]:
                torch.save(name+".pth")
>>>>>>> ea4937ae0ebd9e3cf4dc1493f5f1441cbc175325
                break
"""
Kitchen:
# action: batchsize * 10 * 9 (9-dimensional action)
# pad_mask: batchsize * 11   
# states: batchsize * 11 * 60 (60-dimensional states) label - states[:, -1, :]
# skill_z: batchsize * 10 (10-dimensional latent state)
"""
