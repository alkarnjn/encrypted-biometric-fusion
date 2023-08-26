import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import BinaryAUROC
import torch.nn as nn
from pytorch_metric_learning import losses
from pytorch_metric_learning import distances
from tqdm import tqdm


def get_data(adr=None, device=torch.device("cpu"), test=0.8, train=0.2):
    if adr is None:
        adr = {
            'face': "feature-extraction/extractions/VGGFace_vgg_cplfw.txt",
            'id': "feature-extraction/extractions/deep_speaker_librispeech_google_labels.txt",
            'voice': "feature-extraction/extractions/deep_speaker_librispeech_google.txt"
        }
    # read face and voice matrix
    face_mat = torch.from_numpy(np.loadtxt(adr['face'], dtype=np.float32))
    voice_mat = torch.from_numpy(np.loadtxt(adr['voice'], dtype=np.float32))

    # read individual's id as a panda dataframe to be able to create unique id
    df = pd.read_csv(adr['id'], header=None)
    df.columns = ['id']
    # create label for each id
    data_label = torch.from_numpy((df.groupby('id').ngroup() + 1).values)

    # concatenate face and voice [face,voice]
    # for each voice feature we consider two face features
    feature_dim = face_mat.shape[1] + voice_mat.shape[1]
    data_x = torch.empty((voice_mat.shape[0] * 2, feature_dim))
    data_x[:, face_mat.shape[1]:] = voice_mat.repeat_interleave(2, dim=0)
    for i in range(2):
        data_x[i::2, :face_mat.shape[1]] = face_mat[data_label * 2 - 2 + i, :]
    data_label = data_label.repeat_interleave(2)

    # split to train, validate
    gen = torch.Generator().manual_seed(0)
    x_train, x_val = random_split(data_x.to(device), [test, train], generator=gen)
    gen = torch.Generator().manual_seed(0)
    label_train, label_val = random_split(data_label.to(device), [test, train], generator=gen)

    return (x_train, label_train), (x_val[:], label_val[:]), feature_dim

def normalize(x):
    return x / x.norm(dim=1).unsqueeze(1)


class BiometricDataSet(Dataset):
    def __init__(self, x, label):
        super().__init__()
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.x[index], self.label[index]


class BiometricProjector(nn.Module):
    def __init__(self, feature_dim, projected_dim, device=torch.device("cpu")):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, projected_dim, bias=False, device=device)

    def forward(self, x):
        return self.fc1(x)


class TripletLoss:
    def __init__(self, batch_size, lmb, margin, device):
        self.row, self.col = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
        self.lmb = lmb
        self.push = losses.TripletMarginLoss(margin=margin, distance=distances.DotProductSimilarity())

    def __call__(self, x: torch.tensor, label: torch.tensor):
        x = normalize(x)
        mask = (label.unsqueeze(0) == label.unsqueeze(1))[self.row, self.col]
        row_mask = self.row[mask]
        col_mask = self.col[mask]
        pull = 1 - torch.sum(x[row_mask] * x[col_mask]) / len(row_mask)
        push = self.push(x, label)
        return (1 - self.lmb) * push + self.lmb * pull
    
class AUROC:
    def __init__(self,val_label,device,thresholds=128):
        self.row, self.col = torch.triu_indices(len(val_label),len(val_label), offset=1, device=device)
        self.true_label = (val_label.unsqueeze(0) == val_label.unsqueeze(1))[self.row, self.col]
        self.metric = BinaryAUROC(thresholds=thresholds)

    def compute(self,x):
        x = normalize(x)
        score = (x@x.T)[self.row, self.col]
        return self.metric(score,self.true_label)


device = torch.device("cuda")
torch.set_num_threads(32)
out_dim = 32
batch_size = 128
lr = 1e-3
regularization = 1e-5
lmb = 0.1
margin = 0.5
iteration = 25


data_train, data_val, in_dim = get_data(device=device)
data_train = DataLoader(BiometricDataSet(*data_train), batch_size=batch_size, shuffle=True, drop_last=True)
auroc = AUROC(data_val[1],device=device)

model = BiometricProjector(in_dim, out_dim, device=device)
loss_func = TripletLoss(batch_size, lmb, margin, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=regularization)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=iteration)
auroc_val = torch.nan

with torch.no_grad():
    auroc_init = auroc.compute(model(data_val[0]))

for epoch in range(iteration):
    avg_loss = 0
    p_bar = tqdm(data_train)
    for iter, (x, label) in enumerate(p_bar):
        loss = loss_func(model(x), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.detach()
        if not (iter+1)%20:
            with torch.no_grad():
                auroc_val = auroc.compute(model(data_val[0]))
            p_bar.set_description(f"epoch{epoch+1:<3d}  AVGLoss:{avg_loss/len(data_train):<9.6f}    "
                                  f"AUROC:{auroc_val:<9.6f}   AUROC_Initial:{auroc_init:<9.6f}")