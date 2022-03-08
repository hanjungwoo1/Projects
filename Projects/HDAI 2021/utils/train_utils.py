from utils.metric import metric
from utils.cfg import config as cfg
from dataset import HDataset
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os

class training_and_validate():
    def __init__(self, task, network, logger, args):
        train_dataset = HDataset(task, 'train')
        val_dataset = HDataset(task, 'validation')

        self.logger = logger
        self.args = args
        self.batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
        self.val_batch_size = args.val_batch_size if args.val_batch_size is not None else cfg.val_batch_size
        self.epoch = args.epoch if args.epoch is not None else cfg.epoch
        self.trainloader = DataLoader(train_dataset,
            batch_size = self.batch_size,
            shuffle=True,
            collate_fn = train_dataset.collate_fn,
            num_workers = 2
            )
        self.valloader = DataLoader(val_dataset,
            batch_size = self.val_batch_size,
            shuffle=True,
            collate_fn = val_dataset.collate_fn,
            num_workers = 2
            )

        self.checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_{:s}_{:s}.pth'.format(args.task, args.model))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.net = network
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, cfg.lr_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def logger_info(self, s):
        print(s)
        self.logger.info(s)

    def train(self):
        train_batch_num = len(self.trainloader)
        val_batch_num = len(self.valloader)
        best_val_DSC = 0.
        best_val_Jacc = 0.

        for i in range(self.epoch):
            mean_DSC = 0.
            mean_Jacc = 0.
            mean_DSC_val = 0.
            mean_Jacc_val = 0.

            ###training
            self.logger_info('** Epoch {:d} train start **'.format(i+1))
            for j, (batch_data, batch_label) in enumerate(tqdm(self.trainloader, total = train_batch_num)):
            
                self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)

                pred_label = self.net(batch_data)
                compute_loss = metric(pred_label, batch_label)
                loss = compute_loss.Jaccard_loss() + cfg.k * self.criterion(pred_label, batch_label)
                loss.backward()

                mean_DSC += compute_loss.cal_DSC().item()
                mean_Jacc += compute_loss.Jaccard().item()
                self.optimizer.step()

            ###validation
            self.logger_info('** Epoch {:d} validation start **'.format(i+1))
            with torch.no_grad():
                for j, (batch_data, batch_label) in enumerate(tqdm(self.valloader, total = val_batch_num)):
                    batch_data = batch_data.to(self.device)
                    batch_label = batch_label.to(self.device)

                    pred_label = self.net(batch_data)

                    compute_loss_val = metric(pred_label, batch_label)
                    loss = compute_loss_val.total_loss()
                    mean_DSC_val += compute_loss_val.cal_DSC().item()
                    mean_Jacc_val += compute_loss_val.Jaccard().item()

            mean_DSC = mean_DSC / train_batch_num
            mean_Jacc = mean_Jacc / train_batch_num
            mean_DSC_val = mean_DSC_val / val_batch_num 
            mean_Jacc_val = mean_Jacc_val / val_batch_num 

            self.logger_info('*** Results ***')
            self.logger_info('DSC : {:f}'.format(mean_DSC))
            self.logger_info('Jacc : {:f}'.format(mean_Jacc))
            self.logger_info('DSC_val : {:f}'.format(mean_DSC_val))
            self.logger_info('Jacc_val : {:f}'.format(mean_Jacc_val))
            self.scheduler.step() 
            if best_val_DSC < mean_DSC_val :
                best_val_DSC = mean_DSC_val
                best_val_Jacc = mean_Jacc_val
                torch.save({
                            'model_state_dict' : deepcopy(self.net.state_dict()),
                            'optimizer_state_dict' : deepcopy(self.optimizer.state_dict()),
                            'best_DSC_val' : deepcopy(best_val_DSC),
                            'best_Jacc_val' : deepcopy(best_val_Jacc)
                            }, self.checkpoint_path)

        self.logger_info('Best DSC_val : {:0.4f}'.format(best_val_DSC))
        self.logger_info('Best Jacc_val : {:0.4f}'.format(best_val_Jacc))
        

if __name__ == '__main__':
    training = training_and_validate()
    training.train()