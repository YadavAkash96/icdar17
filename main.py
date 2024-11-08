import logging, os, argparse, math, random, re, glob
import pickle as pk
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch,cv2
from torch import optim
import torch.backends.cudnn as cudnn
import h5py
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision

from pytorch_metric_learning import samplers
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, cosine_scheduler
from torch.optim.lr_scheduler import StepLR

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader,array_to_img_loader

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling, GMP, MaxPooling, LSEPooling

from aug import Erosion, Dilation, Opening, Closing
from utils.triplet_loss import TripletLoss
from utils.AdMSoftmaxLoss import AdMSoftmaxLoss
from utils.ArcMarginModel import ArcMarginModel
from utils.ArcMarginModelFocalLoss import ArcMarginModelFocalLoss

from backbone import resnets
from backbone import densenet
from backbone.model import Model
import torch.nn as nn
import torch.multiprocessing as mp

#mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')



def compute_page_features(_features, writer, pages):
    _labels = list(zip(writer, pages))

    labels_np = np.array(_labels)
    features_np = np.array(_features)
    writer = labels_np[:, 0]
    page = labels_np[:, 1]

    page_features = []
    page_writer = []
    for w, p in tqdm(set(_labels), 'Page Features'): # in training dataset we have 3 pages per author/writer, and 40 authors for validation so 40*3 = 120 pages for validation
        idx = np.where((writer == w) & (page == p))
        page_features.append(features_np[idx])
        page_writer.append(w)

    return page_features, page_writer

def encode_per_class(model, args, poolings=[]):
    testset = args['testset']
    ds = WriterZoo.datasets[testset['dataset']]['set'][testset['set']]
    
    #print(os.path.join(WriterZoo.datasets[testset['dataset']]['basepath'],ds['path']))
    #path = ds['path']
    regex = ds['regex']

    pfs_per_pooling = [[] for i in poolings]

    regex_w = regex.get('writer')
    regex_p = regex.get('page')
    # my changes # hardcoded need to change it
    cur_dir = os.path.join(WriterZoo.datasets[testset['dataset']]['basepath'],ds['path'])
    print(f"Validation Dir: {cur_dir}")
    scriptnet = glob.glob(cur_dir + '/*.h5')
    
    #srcs = sorted(list(glob.glob(f'{path}/**/*.png', recursive=True)))
    srcs = sorted(scriptnet)
    logging.info(f'Found {len(srcs)} images')
    #print(f'Found {len(srcs)} images')
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
        writers.append(w)

    torch.cuda.empty_cache()
    pfs_per_pooling = [np.concatenate(pfs) for pfs in pfs_per_pooling]
    
    return pfs_per_pooling, writers

def inference(model, ds, args):
    model.eval()
    #reducing num workers original was 4, changed to 0
    loader = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=args['test_batch_size'])

    feats = []
    pages = []
    writers = []

    for sample, labels in tqdm(loader, desc='Inference'):
        if len(labels) == 3:
            w,p = labels[1], labels[2]
        else:
            w,p = labels[0], labels[1]

        writers.append(w)
        pages.append(p)
        sample = sample.cuda()
        
        with torch.no_grad():
            emb = model(sample)
            emb = torch.nn.functional.normalize(emb)
        feats.append(emb.detach().cpu().numpy())
    
    feats = np.concatenate(feats)
    writers = np.concatenate(writers)
    pages = np.concatenate(pages)   

    return feats, writers, pages

def test(model, logger, args, name='Test'):

    # define the poolings
    sumps, poolings = [], []
    sumps.append(SumPooling('l2', pn_alpha=0.4)) # this is also power normalization followed by l2 normalization
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

            print(f'Fitting PCA done')
            _eval = Retrieval()
            print(f'Calculate mAP..')

            res, _ = _eval.eval(pfs_tf, writer)

            p = f'{pca_dim}' if pca_dim != -1 else 'full'
            meanavp = res['map']

            if meanavp > best_map:
                best_map = meanavp
                best_top1 = res['top1']
                best_pooling = f'{poolings[i]}-{p}'
                pk.dump(pca, open(os.path.join(logger.log_dir, 'pca_densenet_margincrossloss_subcenter_1.pkl'), "wb"))
                
            table.append([f'{poolings[i]}-{p}', meanavp, res['top1']])
            print(f'{poolings[i]}-{p}-{name} MAP: {meanavp}')
            print(f'''{poolings[i]}-{p}-{name} Top-1: {res['top1']}''')

    logger.log_table(table, 'Results', columns)
    logger.log_value(f'Best-mAP', best_map)
    logger.log_value(f'Best-Top1', best_top1)

###########

def get_optimizer(args, model, metric_arcface):
    #optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_arcface.parameters()}],
    #                weight_decay=args['optimizer_options']['wd'])
    
    optimizer = optim.SGD([{'params': model.parameters()}, {'params': metric_arcface.parameters()}],
                          lr=args['optimizer_options']['base_lr'],
                          momentum=args['optimizer_options']['mom'],
                          weight_decay=args['optimizer_options']['wd'])
    #optimizer = optim.SGD(model.parameters(), 
    #                      lr=args['optimizer_options']['base_lr'],
    #                      momentum=0.9,
    #                      weight_decay=args['optimizer_options']['wd'])
    return optimizer

def validate(model, val_ds, args):
    desc, writer, pages = inference(model, val_ds, args)
    print('Inference done')
    pfs, writer = compute_page_features(desc, writer, pages)

    #print(f'page, and writers: {pfs},{writer}')
    norm = 'powernorm'
    pooling = SumPooling(norm)
    descs = pooling.encode(pfs)
    #print(f'validate desc: {desc}')
    _eval = Retrieval()
    res, _ = _eval.eval(descs, writer)
    meanavp = res['map']
    # my changes to check top1%
    print(f'Top-1%: {res["top1"]}')
    return meanavp

# my changes, used for visualization
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
        
        
def train_one_epoch(model, train_ds, arcface_loss, optimizer, scheduler, epoch, args, logger):

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.train()
    model = model.cuda()
    # set up the triplet stuff, this sampler does introduce some kind of shuffle, as it's selecting some samples from same class, that's why train_ds[0] before and after dataloader won't be exactly same
    sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))
    
    #reducing num workers original was 32
    # my changes, setting drop_last = False, earlier was true reason: it's dropping around 50k , sampler also influences batch calculations
    train_triplet_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=0)    
    
    pbar = tqdm(train_triplet_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(train_triplet_loader) 
    logger.log_value('Epoch', epoch, commit=False)
    
    for i, (samples, label) in enumerate(pbar):
        it = iters * epoch + i
        for i, param_group in enumerate(optimizer.param_groups):
            
            if it > (len(scheduler) - 1):
                param_group['lr'] = scheduler[-1]
            else:
                param_group["lr"] = scheduler[it]
            
            if param_group.get('name', None) == 'lambda':
                param_group['lr'] *= args['optimizer_options']['gmp_lr_factor']
        samples = samples.cuda()
        samples.requires_grad=True
        if args['train_label'] == 'cluster':
            l = label[0]
        if args['train_label'] == 'writer':
            l = label[1]

        l = l.cuda()

        emb = model(samples)
        #additional step
        loss = arcface_loss(emb,l)
        #mychanges amsoftmax loss
        
        #loss = triplet_loss(emb, l, emb, l)
        #loss = am_softmax_loss(emb,l)
        logger.log_value(f'loss', loss.item())
        logger.log_value(f'lr', optimizer.param_groups[0]['lr'])
        logging.info('loss: {} learning rate: {}'.format(loss.item(),optimizer.param_groups[0]['lr']))
        
        optimizer.zero_grad()
        loss.backward()
        #optimizer.clip_gradient(args['optimizer_options']['grad_clip'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2) # this line prevent gradient exploding, 2=> means L2 norm used
        optimizer.step()

    # Step the scheduler after every epoch
    #scheduler.step()

    torch.cuda.empty_cache() # free up cached memory, does not affect memory in use
    return model

def train(model, train_ds, val_ds, args, metric_arcface, logger, optimizer):
    print("GPU:{}".format(next(model.parameters()).is_cuda))
    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], epochs, niter_per_ep, warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=0)
    
    best_epoch = -1
    best_map = validate(model, val_ds, args)

    print(f'Val-mAP: {best_map}')
    logger.log_value('Val-mAP', best_map)

    # mychanges
    #loss = AdMSoftmaxLoss(embeddings=6400, num_classes=5000, s=30.0, m=0.2)
    # new changes
    #metric_arcface= ArcMarginModel(in_features=6400, out_features=5000, scale=30.0, margin=0.5)
    
    #loss = TripletLoss(margin=args['train_options']['margin'])
    #print('Using Triplet Loss')
    #print('Using AM-Softmax loss')
    print('Using Additive-Angular-Margin-Model sub-center with Crossentropy')
    #print('Using Additive-Angular-Margin-Model with Focal Loss')
    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, metric_arcface, optimizer, lr_schedule, epoch, args, logger)
        mAP = validate(model, val_ds, args)

        logger.log_value('Val-mAP', mAP)
        print(f'Val-mAP: {mAP}')


        if mAP > best_map:
            best_epoch = epoch
            best_map = mAP
            save_model(model, optimizer, epoch, os.path.join(logger.log_dir, 'densenet_margincrossloss_subcenter_1.pt'))


        if (epoch - best_epoch) > args['train_options']['callback_patience']:
            break

    # load best model
    checkpoint = torch.load(os.path.join(logger.log_dir, 'densenet_margincrossloss_subcenter_1.pt'))
    print(f'''\nLoading model from Epoch {checkpoint['epoch']}''')
    print(f'Best mAP:{best_map}')
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval() 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def prepare_logging(args):
    os.path.join(args['log_dir'], args['super_fancy_new_name'])
    Logger = getLogger(args["logger"])
    logger = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)
    return logger

def train_val_split(dataset, prop = 0.9):
    authors = list(set(dataset.labels['writer']))
    random.shuffle(authors)

    train_len = math.floor(len(authors) * prop)
    train_authors = authors[:train_len]
    val_authors = authors[train_len:]
    
    print(f'{len(train_authors)} authors for training - {len(val_authors)} authors for validation')

    train_idxs = []
    val_idxs = []
    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.get_label(i)[1] # cluster, writer, page
        if w in train_authors:
            train_idxs.append(i)
        if w in val_authors:
            val_idxs.append(i)

    print(f'train len: {len(train_idxs)}, val len: {len(val_idxs)}')
    train = torch.utils.data.Subset(dataset, train_idxs)
    val = torch.utils.data.Subset(dataset, val_idxs)

    return train, val

def main(args):
    #print(args)
    logger = prepare_logging(args)
    logger.update_config(args)
    
    logger.log_dir=r"/home/vault/iwi5/iwi5232h/resources/training_files/"
    print(f"log dir:{logger.log_dir}")
    backbone = getattr(densenet, args['model']['name'], None)(24, 1.0, 0.2)
    #backbone = getattr(resnets, args['model']['name'], None)()
    if not backbone:
        print("Unknown backbone!")
        raise
    print('----------')
    print(f'Using {type(backbone)} as backbone')
    print(f'''Using {args['model'].get('encoding', 'netvlad')} as encoding.''')
    print('----------')

    random = args['model'].get('encoding', None) == 'netrvlad'
    print(f'random is :{random}')
    model = Model(backbone, dim=64, num_clusters=args['model']['num_clusters'], random=random)
    
    #my changes
    #metric_arfacefocalloss = ArcMarginModelFocalLoss(in_features=6400, out_features=5000, scale=64.0, margin=0.3, gamma=2.0, alpha=0.25)
    metric_arfacecrossloss = ArcMarginModel(in_features=6400, out_features=5000, scale=64.0, margin=0.3, sub_centers=3)
    metric_arcface = nn.DataParallel(metric_arfacecrossloss)
    metric_arcface = metric_arcface.cuda()
    
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.train()
    model = model.cuda()
    tfs = []
   
    # this means if args has key "grayscal" then use it's value otherwise use dafault value "None"
    if args.get('grayscale', None):
        tfs.extend([
            transforms.ToTensor(), # this transform image/ndarray to PyTorch tensors and also perform normalization
            transforms.Grayscale(num_output_channels=3)
        ])
    else:
        tfs.append(transforms.ToTensor())
    
    
    if args.get('data_augmentation', None) == 'morph':
        tfs.extend([
            #transforms.RandomApply([Opening()], p=0.50),
            transforms.RandomApply([Dilation()], p=0.4),
            transforms.RandomApply([Erosion()], p=0.4),
            #transforms.RandomApply([Closing()], p=0.50)
        ])  
    
    transform = transforms.Compose(tfs)
    # chain multiple transformation together eg. resizing, normalization, data augmentation etc
    train_dataset = None
    if args['trainset']:
        d = WriterZoo.get(**args['trainset'])
        train_dataset = d.TransformImages(transform=transform).SelectLabels(label_names=['cluster', 'writer', 'page'])
        
    if args.get('use_test_as_validation', False):
        val_ds = WriterZoo.get(**args['testset'])
        if args.get('grayscale', None):
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3)
            ])
        else:
           
            test_transform = transforms.ToTensor()
        val_ds = val_ds.TransformImages(transform=test_transform).SelectLabels(label_names=['writer', 'page'])

        train_ds = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
        val_ds = torch.utils.data.Subset(val_ds, range(len(val_ds)))
    else:
        train_ds, val_ds = train_val_split(train_dataset)
    
    
    optimizer = get_optimizer(args, model,metric_arcface)
    #=====> commented for new Margin loss

    if args['checkpoint']:
        print(f'''Loading model from {args['checkpoint']}''')
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    
    if not args['only_test']:
        print('running only test version:')
        model, optimizer = train(model, train_ds, val_ds, args, metric_arcface, logger, optimizer)
    # testing
    save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'densenet_margincrossloss_subcenter_1.pt'))
    test(model, logger, args, name='Test')
    logger.finish()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/icdar2017.yml')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=1024, type=int,
                        help='seed')
    # used for round 1 --> default = 2174 Test MAP: 0.7121 Top1: 0.8758    
    # used for round 2 --> default = 1024 Test MAP: 0.7132 Top-1: 0.8744
    # used for round 3 --> default = 42  Test MAP: 0.7137 Top-1: 0.8758
    # used for round 4 used custom kfold --> default = 42 Test MAP: 0.7138 Top-1: 0.8730
    
    args = parser.parse_args()
        
    config = load_config(args)[0]

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    
    seed_everything(args.seed)
    
    main(config)