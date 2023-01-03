from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GMMConv, GCNConv, MessagePassing
from torch_geometric.nn import Linear, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops, scatter
import torch
from torch.nn import BatchNorm1d, Dropout, SELU
from datamodule.datasets.kellerdataset import KellerDataset
from datamodule.datasets.dravnieks_dataset import DravnieksDataset
from datamodule.datasets.musk_dataset import MuskDataset
from datamodule.musk_cv_datamodule import MuskDataModule
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_accuracy
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm
import numpy as np
import pandas as pd

torch.manual_seed(42)
random_seed = 42
# traindataset = DravnieksDataset()
# testdataset = KellerDataset()
dataset = MuskDataset()

data = "musk"

def split_geometric(dataset, K = 5, train_size = 0.75):
    X = np.zeros((len(dataset), dataset[0].x.shape[1]))
    label = torch.tensor([])
    for data in dataset:
        label = torch.cat((label, data.y[:,1]),0)
    if K>=2:
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_seed)
        return (skf.split(X=X, y=label))
    elif K==1:
        sss = StratifiedShuffleSplit(n_splits=K, random_state= random_seed, train_size=train_size)
        return sss.split(X=X, y=label)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,pool_dim, fully_connected_channels):
        super(GCN, self).__init__()
        torch.manual_seed(433)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = GCNConv(hidden_channels[1], hidden_channels[2])
        self.conv4 = GCNConv(hidden_channels[2], hidden_channels[3])
        # self.conv5 = GCNConv(hidden_channels[3], pool_dim)
        self.lin1 = Linear(pool_dim, fully_connected_channels[0])
        self.lin2 = Linear(fully_connected_channels[0], fully_connected_channels[1])
        self.lin3 = Linear(fully_connected_channels[1], fully_connected_channels[2])

        self.lin = Linear(fully_connected_channels[2], dataset.num_classes)

        self.lin_1 = Linear(hidden_channels[0],pool_dim)
        self.lin_2 = Linear(hidden_channels[1],pool_dim)
        self.lin_3 = Linear(hidden_channels[2],pool_dim)
        self.lin_4 = Linear(hidden_channels[3],pool_dim)

        self.norm1 = BatchNorm1d(fully_connected_channels[0])
        self.norm2 = BatchNorm1d(fully_connected_channels[1])
        self.norm3 = BatchNorm1d(fully_connected_channels[2])

        self.drop = Dropout(p=0.47)
        self.activate = SELU()

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        readout=0

        x = self.conv1(x, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_1(x), dim=-1)

        x = self.conv2(x, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_2(x), dim=-1)
        
        x = self.conv3(x, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_3(x), dim=-1)

        x = self.conv4(x, edge_index)
        x = self.activate(x)
        x = self.max_graph_pool(x, edge_index)
        readout += F.softmax(self.lin_4(x), dim=-1)

        # 2. Readout layer
        x = global_add_pool(readout, batch)  # [batch_size, hidden_channels]
        # x = F.softmax(x, dim=1)

        # 3. Fully Connected Neural Net
        x = self.lin1(x)
        x = x.relu()
        x = self.norm1(x)
        x = self.drop(x)

        x = self.lin2(x)
        x = x.relu()
        x = self.norm2(x)
        x = self.drop(x)

        x = self.lin3(x)
        x = x.relu()
        x = self.norm3(x)
        x = self.drop(x)

        ## prediction phase
        x = self.lin(x)
        x = x.sigmoid()
    
        return x
    
    @staticmethod
    def max_graph_pool(x, edge_index):
        edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        x = scatter(x[row], col, dim=0, reduce="max")
        return x

model = GCN(hidden_channels=[15,20,27,36], pool_dim=175, fully_connected_channels=[84, 42, 16])
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

def train(loader):
    model.train()
    for i, data in enumerate(loader,0):  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out[:,1].float(), data.y[:,1].float()) # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
    scheduler.step()

def test(loader):
    model.eval()
    pred = torch.tensor([])
    y_truth = torch.tensor([])
    out_proba = torch.tensor([])
    with torch.no_grad():
        for i, data in enumerate(loader,0):  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            out = (F.softmax(out, dim=1))
            if len(pred)==0:
                # pred = (out>0.5).astype(int)
                out_proba = out
                y_truth = data.y
            else:
                # pred = torch.vstack([pred,(out>0.5).astype(int)])
                y_truth = torch.vstack([y_truth,data.y])
                out_proba = torch.vstack([out_proba,out])
        _, pred = torch.max(out_proba, 1)
        y = (y_truth[:,1])
    return  binary_accuracy(pred,y).item(), multiclass_auroc(out_proba, y_truth[:,1], num_classes=2).item(), criterion(out[:,1].float(), data.y[:,1].float()).item()

if __name__=="__main__":
    # column = pd.read_csv(f"data/{data}/raw/{data}_label_data.csv").iloc[:,4:].columns
    num_epoch = 300
    K = 5
    n_epochs_stop = 100

    df = pd.DataFrame(index=range(K), columns=["train", "test"])
    train_data = dataset[:47]
    test_data = dataset[47:]
    
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    split = split_geometric(dataset=train_data,K=K)
    for i, (train_idx, val_idx) in enumerate(split):
        val_idx=train_idx
        min_val_loss = np.inf
        train_dataset = train_data[train_idx]
        val_dataset = train_data[val_idx]

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        for epoch in tqdm(range(1, num_epoch)):
            train(train_loader)
            _,_,val_loss = test(val_loader)

            if val_loss < min_val_loss:
                # print(min_val_loss)
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                # print(epochs_no_improve)
            
            if epoch > 5 and epochs_no_improve == n_epochs_stop:
                early_stop = True
                train_acc, train_auroc,_ = test(train_loader)
                break
            elif epoch+1==num_epoch:
                train_acc, train_auroc,_ = test(train_loader)
            else:
                continue
            
        
        acc, auroc, _ = test(test_loader)


        df["train"].loc[i] = (train_acc, train_auroc)
        df["test"].loc[i] = (acc, auroc)
        # print(df)

    # df.to_csv(f"outputs/performance_{data}.csv")
    print(df)