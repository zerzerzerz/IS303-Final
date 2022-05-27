from turtle import forward
import torch
from torch import nn

class MLP(nn.Module):
    """vanilla MLP"""
    def __init__(self,input_dim, num_embedding, embedding_dim, hidden_dim, num_layer:int, dropout:float=0.2, output_dim:int=1, use_bn:bool=True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bn = use_bn
        self.num_layer = num_layer
        self.dropout = dropout
    
        self.title_embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        model = []
        self.real_input_dim = self.input_dim  + self.embedding_dim

        model.append(nn.Linear(self.real_input_dim,self.hidden_dim))
        model.append(nn.LeakyReLU(0.2))
        if self.use_bn:
            model.append(nn.BatchNorm1d(hidden_dim))
        model.append(nn.Dropout(self.dropout))

        for _ in range(self.num_layer):
            model.append(nn.Linear(self.hidden_dim,self.hidden_dim))
            model.append(nn.LeakyReLU(0.2))
            if self.use_bn:
                model.append(nn.BatchNorm1d(hidden_dim))
            model.append(nn.Dropout(self.dropout))

        model.append(nn.Linear(self.hidden_dim, self.output_dim))
        model.append(nn.Softplus())

        self.model = nn.Sequential(*model)
        self.loss_fn = nn.MSELoss(reduction='mean')
    

    def forward(self,feature,title,price):
        """
        Input:
            feature: shape = [B,D]
            title: shape = [B]
            price: shape = [B]
        Return:
            price_pred: shape = [B]
            loss: scalar 
        """
        title_embedding = self.title_embedding(title)
        feature_cat = torch.cat([title_embedding, feature],dim=1)
        price_pred = self.model(feature_cat).squeeze(-1)
        loss = self.loss_fn(price_pred,price)
        return price_pred, loss


class SkipConnectionBlock(nn.Module):
    def __init__(self,dim,use_bn,dropout) -> None:
        super().__init__()
        self.dim = dim
        self.use_bn = use_bn
        self.dropout = dropout
        skip_block = []
        
        skip_block.append(nn.Linear(self.dim,self.dim))
        skip_block.append(nn.LeakyReLU(0.2))
        if self.use_bn:
            skip_block.append(nn.BatchNorm1d(self.dim))
        skip_block.append(nn.Dropout(self.dropout))
        self.skip_block = nn.Sequential(*skip_block)
    
    def forward(self, feature_cat):
        return feature_cat + self.skip_block(feature_cat)


class MLP2(MLP):
    """Skip-connection MLP"""
    def __init__(self, input_dim, num_embedding, embedding_dim, hidden_dim, num_layer: int, dropout: float = 0.2, output_dim: int = 1, use_bn: bool = True) -> None:
        super().__init__(input_dim, num_embedding, embedding_dim, hidden_dim, num_layer, dropout, output_dim, use_bn)
        del self.model

        model = []
        self.real_input_dim = self.input_dim  + self.embedding_dim

        model.append(nn.Linear(self.real_input_dim,self.hidden_dim))
        model.append(nn.LeakyReLU(0.2))
        if self.use_bn:
            model.append(nn.BatchNorm1d(hidden_dim))
        model.append(nn.Dropout(self.dropout))


        for _ in range(self.num_layer):
            model.append(SkipConnectionBlock(self.hidden_dim, self.use_bn, self.dropout))

        model.append(nn.Linear(self.hidden_dim, self.output_dim))
        model.append(nn.Softplus())

        self.model = nn.Sequential(*model)
    

        


if __name__ == "__main__":
    input_dim = 16
    num_embedding = 50 
    embedding_dim = 8
    hidden_dim = 32 
    num_layer = 2
    batch_size = 128

    model = MLP(input_dim, num_embedding, embedding_dim, hidden_dim, num_layer)
    feature = torch.rand((batch_size, input_dim))
    title = torch.randint(0,num_embedding,(batch_size, ))
    price = torch.rand((batch_size, ))

    price_pred, loss = model(feature, title, price)
    pass