import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self,csv_path='data/data.csv') -> None:
        super().__init__()
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.features = []
        self.prices = []
        for index in df.index:
            price = df.loc[index,'price']
            feature = torch.from_numpy(
                    df.loc[index,[
                    "change-speed",
                    "kilometer",
                    "shangpai-date",
                    "standard",
                    "title"
                ]].to_numpy()
            ).to(torch.float32)
            self.features.append(feature)
            self.prices.append(price)
            
        
        # normalize
        self.price_mean = torch.Tensor(self.prices).mean()
        self.price_std = torch.Tensor(self.prices).std()
        self.feature_mean = torch.cat(self.features).mean(dim=0)
        self.feature_std = torch.cat(self.features).std(dim=0)

        self.eps = 1e-12
        for i in range(len(self.prices)):
            self.prices[i] = (self.prices[i] - self.price_mean) / (self.price_std + self.eps)
        for i in range(len(self.features)):
            self.features[i] = (self.features[i] - self.feature_mean) / (self.feature_std + self.eps)

        self.length = len(self.prices)

    def __getitem__(self, index):
        return self.features[index], self.prices[index]
    
    
    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = MyDataset()
    dataloader = DataLoader(dataset,2)

    for x,y in dataloader:
        print(x.shape)
        print(y.shape)
        break