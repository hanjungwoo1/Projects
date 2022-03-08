from dataset import HDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import logging
import os
from tqdm import tqdm
from PIL import Image

log_dir = 'log'

class data_logger():
    def __init__(self, task, mode):
        ### log initialize ###
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_fname = os.path.join(log_dir, 'data_log_{:s}_{:s}.txt'.format(task, mode))
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger(__name__)
        self.datasize_type = { 0 : (434, 636),
            1 : (422, 636),
            2 : (600, 800),
            3 : (720, 960),
            4 : (768, 1024)
            }


        self.task = task
        self.mode = mode
        self.dataset = HDataset(task, mode)

    def datalist_log(self, ):
        self.logger.info('**** logging start ****')
        size = [0] * (len(self.datasize_type.keys()) + 1)
        min_dsize = [10000,10000]
        max_dsize = [0,0]
        for i in range(self.dataset.__len__()):
            data, label = self.dataset.__getitem__(i)
            self.logger.info('{:04d} Data  Shape: {:d} x {:d} x {:d}'.format(i+1, data.shape[0], data.shape[1], data.shape[2]))
            self.logger.info('{:04d} Label Shape: {:d} x {:d} x {:d}'.format(i+1, label.shape[0], label.shape[1], label.shape[2]))


            ### data size ###
            etc = True
            for t, value in self.datasize_type.items():
                if data.shape[1:] == value :
                    etc = False
                    size[t] += 1
            if etc : size[-1] += 1
                
            for i in range(2):
                if min_dsize[i] > data.shape[i+1] :
                    min_dsize[i] = data.shape[i+1]
                if max_dsize[i] < data.shape[i+1] :
                    max_dsize[i] = data.shape[i+1]

        self.logger.info('**** Result ****')
        self.logger.info('Dataset Name : {:s}'.format(self.task))
        self.logger.info('Dataset Mode : {:s}'.format(self.mode))
        self.logger.info('Statistics')
        for i in range(6) :
            if i < 5 :
                i_size = self.datasize_type[i]
                self.logger.info('Size {:d} x {:d} : {:d}'.format(i_size[0], i_size[1], size[i]))
            else :
                self.logger.info('Etc : {:d}'.format(size[i]))
        self.logger.info('Min size : {:d} x {:d}'.format(min_dsize[0], min_dsize[1]))
        self.logger.info('Max size : {:d} x {:d}'.format(max_dsize[0], max_dsize[1]))

        self.logger.info('**** logging end ****')

if __name__ == '__main__':
    dataset_for = HDataset('A4C', 'train')

    dataloader_for = DataLoader(dataset_for,
            batch_size = 10,
            shuffle=True,
            collate_fn = dataset_for.collate_fn,
            num_workers = 0
            )

    print(len(dataloader_for))

    for i, (batch_data, batch_label) in enumerate(tqdm(dataloader_for, total = len(dataloader_for))):
        if i % 10 == 0 : 
            print(torch.min(batch_data[0]), torch.max(batch_data[0]))
            save_image( batch_label[0] * 255, '{:d}.png'.format(i))

    


    
    
    
    
    
    