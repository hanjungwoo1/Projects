from utils.metric import metric
from utils.cfg import config as cfg
from dataset import HDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

class tester():
    def __init__(self, task, network, logger, args):
        test_dataset = HDataset(task, 'test')


        self.logger = logger
        self.args = args
        self.batch_size = args.batch_size if args.batch_size is not None else cfg.val_batch_size
        self.num_vote = args.num_vote if args.num_vote is not None else cfg.num_vote
        self.testloader = DataLoader(test_dataset,
            batch_size = self.batch_size,
            shuffle=True,
            collate_fn = test_dataset.collate_fn,
            num_workers = 2
            )

        self.checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_{:s}_{:s}.pth'.format(args.task, args.model))
        self.checkpoint = torch.load(self.checkpoint_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.net = network
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.net.to(self.device)
        self.logger_info('Loading Checkpoint Network is Successful')
        self.logger_info('Task : {:s}'.format(task))
        self.logger_info('Model : {:s}'.format(args.model))
        self.logger_info('Best Validation DCS : {:0.4f}'.format(self.checkpoint['best_DSC_val']))
        self.logger_info('Best Validation Jaccard : {:0.4f}'.format(self.checkpoint['best_Jacc_val']))

    def logger_info(self, s):
        print(s)
        self.logger.info(s)

    def test(self):
        test_batch_num = len(self.testloader)

        final_DSC = 0.
        final_Jacc = 0.

        for i in range(self.num_vote):
            mean_DSC = 0.
            mean_Jacc = 0.

            self.logger_info('** Vote {:d} test start **'.format(i+1))

            ###test
            with torch.no_grad():
                for j, (batch_data, batch_label) in enumerate(tqdm(self.testloader, total = test_batch_num)):
                    batch_data = batch_data.to(self.device)
                    batch_label = batch_label.to(self.device)

                    pred_label = self.net(batch_data)

                    compute_loss = metric(pred_label, batch_label)
                    mean_DSC += compute_loss.cal_DSC().item()
                    mean_Jacc += compute_loss.Jaccard().item()

            mean_DSC = mean_DSC / test_batch_num
            mean_Jacc = mean_Jacc / test_batch_num
            final_DSC += mean_DSC
            final_Jacc += mean_Jacc

            if (i + 1) % (self.num_vote // 10) == 0 :
                self.logger_info('*** {:d} Vote Results ***'.format(i+1))
                self.logger_info('DSC : {:f}'.format(mean_DSC))
                self.logger_info('Jacc : {:f}'.format(mean_Jacc))
        
        self.logger_info('*** Final Test Results ***')
        self.logger_info('DSC_val : {:f}'.format(final_DSC / self.num_vote))
        self.logger_info('Jacc_val : {:f}'.format(final_Jacc / self.num_vote))
        