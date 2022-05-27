from torch.optim import Adam
from tqdm import tqdm
from model.model import MLP2 as Model
import torch
from utils.utils import load_json, Logger, save_json, mkdir, setup_seed
from argparse import ArgumentParser
from dataset.dataset import MyDataset
from torch.utils.data import DataLoader
from os.path import join, isfile
from matplotlib import pyplot as plt


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='data/data_train.csv')
    parser.add_argument('--test_dataset', type=str, default='data/data_test.csv')
    parser.add_argument('--output_dir', type=str, default='result/debug-0527')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--save_ck_epoch_gap', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--only_eval', action='store_true')


    # model settings
    parser.add_argument('--input_dim', type=int, default=4)
    parser.add_argument('--num_embedding', type=int, default=len(load_json('data/car_type_refinement.json')))
    parser.add_argument('--embedding_dim', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    args = parser.parse_args()
    args.train = not args.only_eval
    args.num_epoch = args.num_epoch if args.train else 1

    return args


def get_dataloader(args):
    train = MyDataset(args.train_dataset)
    test = MyDataset(args.test_dataset)
    train_loader = DataLoader(train,args.batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoader(test,args.batch_size,shuffle=False,drop_last=True)
    return train_loader, test_loader


def get_model(args):
    ckp = args.checkpoint
    if ckp == '' or ckp is None:
        pass
    elif isfile(ckp):
        return torch.load(ckp)
    else:
        pass

    return Model(args.input_dim, args.num_embedding, args.embedding_dim, args.hidden_dim, args.num_layer, args.dropout)


def save_model(model,path,device):
    torch.save(model.cpu(), path)
    model.to(device)


def run(mode,epoch,model,dataloader,optimizer,ck_dir,logger,args):
    enable_grad = (mode == "train")
    with torch.set_grad_enabled(enable_grad):
        epoch_loss = 0
        for count, (feature, title, price) in tqdm(enumerate(dataloader)):
            feature = feature.to(args.device)
            title = title.to(args.device)
            price = price.to(args.device)
            price_pred, loss = model(feature, title, price)
            if mode == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= (count+1)
        logger.log('Epoch=%d, loss=%.6f'%(epoch,epoch_loss))
        if epoch % args.save_ck_epoch_gap == 0:
            save_model(model, join(ck_dir,f'epoch_{epoch}.pt'), args.device)
        return epoch_loss


if __name__ == "__main__":
    args = get_args()
    setup_seed(args.seed)

    # mkdir
    base = args.output_dir
    mkdir(base)
    ck_dir = join(base,'ck')
    mkdir(ck_dir)
    fig_dir = join(base, 'fig')
    mkdir(fig_dir)

    # save config
    save_json(vars(args), join(base, 'args.json'))

    # create logger
    test_logger = Logger(join(base,'test.txt'))
    train_logger = Logger(join(base,'train.txt')) if args.train else None

    # create loader
    test_logger.log("Loading dataset...")
    train_loader, test_loader = get_dataloader(args)

    # create model
    test_logger.log("Loading model...")
    model = get_model(args)
    model.to(args.device)
    test_logger.log(str(model))

    # create optimizer
    optimizer = Adam(model.parameters(),lr=args.lr) if args.train else None

    # run epochs
    best_eval_loss = 1e12
    loss_record = {
        "train": [],
        "eval": []
    }
    for epoch in tqdm(range(1,args.num_epoch+1)):
        if args.train:
            train_loss = run("train", epoch, model, train_loader, optimizer, ck_dir, train_logger, args)
            loss_record['train'].append(train_loss)

        eval_loss = run("test", epoch, model, test_loader, None, ck_dir, test_logger, args)
        loss_record['eval'].append(eval_loss)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_model(model, join(ck_dir,'best.pt'), args.device)

    # save
    plt.plot(loss_record['train'],c='r',label='train')
    plt.plot(loss_record['eval'],c='b',label='eval')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(which='both')
    plt.savefig(join(fig_dir, 'loss.png'))
    plt.close()
    save_json(loss_record, join(base, 'loss.json'))
    save_model(model, join(ck_dir,'final.pt'), args.device)

