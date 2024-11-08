# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 22:17:29 2024

@author: Akash
"""
import logging, os, argparse, math, re, glob
import pickle as pk
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn
import h5py
from utils.utils import GPU, seed_everything, load_config
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, cosine_scheduler

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader,array_to_img_loader

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling, GMP, MaxPooling, LSEPooling

from utils.triplet_loss import TripletLoss

from backbone import resnets
from backbone.model import Model


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        

def encode_per_class(model, args, poolings=[]):
    testset = args['testset']
    ds = WriterZoo.datasets[testset['dataset']]['set'][testset['set']]
    path = ds['path']
    regex = ds['regex']

    pfs_per_pooling = [[] for i in poolings]

    regex_w = regex.get('writer')
    regex_p = regex.get('page')
    # my changes
    cur_dir = args['data_dir']
    scriptnet = glob.glob(cur_dir + '/*.h5')
    
    #srcs = sorted(list(glob.glob(f'{path}/**/*.png', recursive=True)))
    srcs = sorted(scriptnet)
    logging.info(f'Found {len(srcs)} images')
    print(f'Found {len(srcs)} images')
    writer = [int('_'.join(re.search(regex_w, Path(f).name).groups())) for f in srcs]
    page = [int('_'.join(re.search(regex_p, Path(f).name).groups())) for f in srcs]

    labels = list(zip(writer, page))

    if args.get('grayscale', None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ])
    else:
        transform = transforms.ToTensor()

    np_writer = np.array(writer)
    np_page = np.array(page)

    print(f'Found {len(list(set(labels)))} pages.')

    writers = []
    for w, p in tqdm(set(labels), total=len(set(labels)), desc='Page Features'):
        idx = np.where((np_writer == w) & (np_page == p))[0]
        fps = [srcs[i] for i in idx]
        #my changes
        all_images = []
        for filen in tqdm(fps, total=len(fps)):
            h5f = h5py.File(filen,'r')
            all_images.extend([np.array(v) for v in h5f.values()])
        
        #my changes            
        #ds = FilepathImageDataset(fps, pil_loader, transform)
        ds = FilepathImageDataset(all_images, array_to_img_loader, transform)
        # this is more precise method
        #print(f'ds=>>>>>> {ds}')
        #reducing num workers original was 4
        loader =  torch.utils.data.DataLoader(ds, num_workers=0, batch_size=args['test_batch_size'])
        
        val_loader = iter(loader)
        images = next(val_loader)
        
        img_grid = torchvision.utils.make_grid(images[:4])
        print(f'Shape of images after dataloader retrivel :{images.shape}')
        print(f'dataset size after transformation applied:{len(loader.dataset)}')
        
        #matplotlib_imshow(img_grid, one_channel=False)
        
        feats = []
        for img in loader:
            #print(img.shape)
            img = img.cuda()

            with torch.no_grad():
                feat = model(img)
                feat = torch.nn.functional.normalize(feat)
            feats.append(feat.detach().cpu().numpy())

        feats = np.concatenate(feats)

        for i, pooling in enumerate(poolings):
            enc = pooling.encode([feats])
            pfs_per_pooling[i].append(enc)
        print(f'size of pfs per pooling:{len(pfs_per_pooling)}')
        print(f'value in pfs pooling:{len(pfs_per_pooling[0])}')
        writers.append(w)

    torch.cuda.empty_cache()
    pfs_per_pooling = [np.concatenate(pfs) for pfs in pfs_per_pooling]
    print(f'value of pfs_per_pooling=>>>>>>>>>{pfs_per_pooling[0].shape}')
    return pfs_per_pooling, writers

def test(model, logger, args, name='Test'):

    # define the poolings
    sumps, poolings = [], []
    sumps.append(SumPooling('l2', pn_alpha=0.4))
    poolings.append('SumPooling-PN0p4')

    # extract the global page descriptors
    pfs_per_pooling, writer = encode_per_class(model, args, poolings=sumps)

    best_map = -1
    best_top1 = -1
    best_pooling = ''

    table = []
    columns = ['Pooling', 'mAP', 'Top1']
    for i, pfs in enumerate(pfs_per_pooling):

        # pca with whitening and l2 norm
        for pca_dim in [512]:

            if pca_dim != -1:
                pca_dim = min(min(pfs.shape), pca_dim)
                print(f'Fitting PCA with shape {pca_dim}')

                pca = PCA(pca_dim, whiten=True)
                pfs_tf = pca.fit_transform(pfs)
                pfs_tf = normalize(pfs_tf, axis=1)
            else:
                pfs_tf = pfs

            print('Fitting PCA done')
            _eval = Retrieval()
            print('Calculate mAP..')

            res, _ = _eval.eval(pfs_tf, writer)

            p = f'{pca_dim}' if pca_dim != -1 else 'full'
            meanavp = res['map']

            if meanavp > best_map:
                best_map = meanavp
                best_top1 = res['top1']
                pk.dump(pca, open(os.path.join(args['output_dir'], 'pca.pkl'), "wb"))
                
            table.append([f'{poolings[i]}-{p}', meanavp, res['top1']])
            print(f'{poolings[i]}-{p}-{name} MAP: {meanavp}')
            print(f'''{poolings[i]}-{p}-{name} Top-1: {res['top1']}''')

    #logger.log_table(table, 'Results', columns)
    print(f'{table}, Results, {columns}')
    #logger.log_value(f'Best-mAP', best_map)
    print(f'Best-mAP: {best_map}')
    #logger.log_value(f'Best-Top1', best_top1)
    print(f'Best-Top1: {best_top1}')
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=r'config/icdar2017.yml', help='Path to config file')
    parser.add_argument('--checkpoint', default = r'None', help='Path to model checkpoint')
    parser.add_argument('--data_dir', default=r'None', help='Path to directory containing h5 files')
    parser.add_argument('--output_dir',default= r'None', help='Directory to save PCA and other results')
    args = parser.parse_args()
    
    args = load_config(args)[0]
    
    
    checkpoint = torch.load(args['checkpoint'])
    random = args['model'].get('encoding', None) == 'netrvlad'
    backbone = getattr(resnets, args['model']['name'], None)()
    model = Model(backbone, dim=64, num_clusters=args['model']['num_clusters'], random=random)
    model = model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test(model, None, args)
    #sumps, poolings = [], []
    #sumps.append(SumPooling('l2', pn_alpha=0.4))
    #poolings.append('SumPooling-PN0p4')
    #p,w = encode_per_class(model, args,sumps)