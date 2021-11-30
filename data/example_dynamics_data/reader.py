import torch, os
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
# actions (sustitute the following code with your directory of example_dynamics_data)
"""
for d in os.listdir("/home/kaiyan3/ECE598SG-project/data/example_dynamics_data/kitchen"):
    if d.find(".pt") != -1:
        print(d)
        data = torch.load("/home/kaiyan3/ECE598SG-project/data/example_dynamics_data/kitchen/"+d)
        print(data.shape) 
"""
torch.manual_seed(21974921)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="0 is direct, 1 is LSTM", default=0,type=int)
    parser.add_argument("--mode2", help="q or qhat", default="q",type=str)
    parser.add_argument("--name", help="name of environment", default="office", type=str)
    parser.add_argument("--path", help="path of data", default= "data/example_dynamics_data/")
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
    # idx = torch.randperm(n)
    # FIXME: remember to shuffle the data!!!
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
    
    def predict(self, x):
        return self.forward(x)


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
    
    def predict(self, x):
        return self.forward(x)[:, -1, :]


if __name__ == "__main__":
    
    
    
    args = get_args()
    path = args.path +args.name+"/" # change this to your directory!  
    device = torch.device("cuda:0")
    
    
    
    mode2 = args.mode2
    # states = torch.load(path+"states0.pt", map_location=device)
    # skill_z = torch.load(path+"skill_z0.pt", map_location=device)
    states = torch.load(path+"states0_"+mode2+".pt", map_location=device)
    skill_z = torch.load(path+"skill_z0_"+mode2+".pt", map_location=device)

    # print(torch.norm(states[0, 0, :]))
    """
    print(states[0, 0, :], skill_z[0, :], states[0, -1, :])
    
    print("pred:", pred, "error:", nn.MSELoss()(pred, ))
    
    pred: tensor([[-1.3050e+00, -1.3228e+00,  1.5239e+00, -2.2134e+00,  3.3100e-01,
          1.9347e+00,  1.6358e+00,  2.4981e-02,  3.3287e-02,  1.1228e-02,
         -1.0135e-03, -3.4664e-01,  2.5808e-04,  5.3097e-03,  1.6267e-03,
         -2.5658e-02,  3.0641e-03, -4.1701e-02, -1.8471e-02, -6.0221e-03,
          7.2722e-03, -9.7901e-03, -7.0847e-01, -2.6956e-01,  3.3178e-01,
          1.5825e+00,  9.7607e-01,  5.9867e-03,  6.6576e-04, -2.1983e-02,
         -4.0475e-03, -6.8757e-04,  4.0796e-03,  2.7309e-03,  7.3015e-03,
         -3.6760e-03, -8.5477e-03, -2.6050e-03, -5.5450e-04,  9.6561e-03,
         -2.1871e-03, -8.5094e-01, -8.9897e-03, -3.4020e-04, -3.7458e-03,
          5.0612e-04, -3.6187e-04, -6.7823e-01, -4.4577e-02,  6.1427e-03,
          9.2872e-03, -2.7959e-03, -7.3320e-01, -2.1511e-01,  7.2636e-01,
          1.5804e+00,  9.6519e-01, -3.3352e-03, -2.6872e-03, -5.3367e-02]],
       device='cuda:0', grad_fn=<AddmmBackward>)
    """
    # exit(0)
    if args.mode == 0:
        feature = torch.cat((states[:, 0, :], skill_z), dim=1) # 12325 * 70
        label = states[:, -1, :] # 12325 * 60
        net = PlainNet(feature.shape[1], label.shape[1]).to(device)
    else:   # alternative: given s_t and skill_z, use a LSTM to predict s_{t+1} to s_{t+N}
        feature = torch.cat((states[:, 0, :], skill_z), dim=1)
        label = states[:, 1:, :] # 12325 * 10 * 60
        net = LSTM(feature.shape[1], label.shape[-1]).to(device)
    
    mix_with_newdata = "no" # True
    
    
    
    if mix_with_newdata in ["true","only"]:
        if args.mode == 0:
            states1 = torch.load(path+"states_prior_1.pt", map_location=device)
            states2 = torch.load(path+"states_prior_2.pt", map_location=device)
            skill_z1 = torch.load(path+"skill_z_prior_1.pt", map_location=device)
            print("states1.shape:", states1.shape)
            print("states2.shape:", states2.shape)
            print("skill_z1.shape:",skill_z1.shape)
            if mix_with_newdata == "true":
                feature = torch.cat([feature, torch.cat([states1, skill_z1], dim=1)], dim=0) # TODO:加限制
                label = torch.cat([label, states2], dim=0)
            else:
                feature = torch.cat([states1, skill_z1], dim=1)
                label = states2
        else: raise NotImplementedError("not implemented!")
    # we are currently not supporting mix_with_newdata and LSTM together.
    
    # print(feature[0].view(1, -1) - torch.cat([zustand, fahigkeit], axis=1), label[0].view(1, -1) - aktion)
    # exit(0)
    
    train_loader, test_loader = get_traj_dataset(feature, label)
    optimizer = torch.optim.Adam(net.parameters())
    N = 15000 # for kitchen
    train_losses, valid_losses = [], []
    for epoch in range(N):
        net.train() 
        avg_loss = 0
        for batch_idx, samples in enumerate(train_loader, 0):
            feature, label = samples["feature"], samples["label"]
            optimizer.zero_grad()
            outputs = net(feature)
            loss = ((outputs-label) ** 2).sum(dim=1).mean()
            loss.backward() 
            optimizer.step()
            avg_loss += loss.item()
        train_losses.append(avg_loss / len(train_loader))
        print("epoch", epoch, "training loss:", avg_loss / len(train_loader))
        net.eval()
        
        for batch_idx, samples in enumerate(test_loader, 0):
            feature, label = samples["feature"], samples["label"]
            outputs = net(feature)
            loss = ((outputs-label) ** 2).sum(dim=1).mean()
            
            if args.mode == 0: losses = ((outputs-label) ** 2).sum(dim=1)
            else: 
                losses = ((outputs - label) ** 2).sum(dim=1).sum(dim=1)
            # plt.hist(losses.detach().cpu().numpy(), bins=40)
            # plt.savefig("fig/fig_epoch_"+str(epoch)+"_mode_"+str(args.mode)+".png")
            # plt.cla()
            
            valid_losses.append(loss.cpu().detach())
            # if epoch == N - 1: print(outputs[0], label[0])
            print("test loss:", loss)
        if len(valid_losses) > 100:  # early stoppingc
            if valid_losses[-1] > valid_losses[-2] and valid_losses[-2] > valid_losses[-3] and valid_losses[-3] > valid_losses[-4]:
                mode = "MLP" if args.mode == 0 else "LSTM"
                torch.save(net, args.path+"models/"+args.name+"/"+mode+"_"+mode2+"_"+mix_with_newdata+".pth")
                plt.plot([i for i in range(len(valid_losses))], valid_losses)
                plt.plot([i for i in range(len(train_losses))], train_losses)
                plt.legend(["valid", "train"])
                plt.yscale("log")
                plt.savefig(args.path+"models/"+args.name+"/"+mode+"_"+mode2+"_"+mix_with_newdata+".jpg")
                exit(0)
                # print("prediction:", net(torch.cat([zustand, fahigkeit], axis=1)))
                break

"""
Kitchen:
# action: batchsize * 10 * 9 (9-dimensional action)
# pad_mask: batchsize * 11   
# states: batchsize * 11 * 60 (60-dimensional states) label - states[:, -1, :]
# skill_z: batchsize * 10 (10-dimensional latent state)
"""
