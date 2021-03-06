# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
import pdb
import hnswlib
import numpy as np
import time
from collections import defaultdict
from tensorboardX import SummaryWriter
import utils
import utils.optimizers as optimizers
import models
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
pretrain_epochs = 1
batch_size = 32
iteration = 10000
lr = [0.001, 0.01]
LEARNING_RATE = 0.001
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/root/share/Original_images/"
source1_name = "webcam"
source2_name = 'dslr'
sources = ["webcam", "dslr"]
original_target_name = "amazon"
dataset = "office31"
    
# Target Hard
hard_threshold = 0.5
    
# HNSW
k = 100
ef = 60
ef_construction = 200
num_elements = 5000
dim = 512
M = 16
index = hnswlib.Index(space='l2', dim=dim) # dim 向量维度
    
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}


def pretrain(model, optimizer, source_name):
    if dataset == "office31":
        source_name = source_name + '/images/'
    source1_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
    source1_iter = iter(source1_loader)
    iters = 0
    iteration = len(source1_loader)
    for ep in range(pretrain_epochs):
        for i, data in enumerate(source1_loader):
            source_data, source_label = data

            optimizer.zero_grad()

            loss = model(None,
                   None,
                   [source_data, source_label],
                   None,
                   None,
                   meta_train=False, mark=0)
            loss.backward()
            optimizer.step()

            if (i + iters) % log_interval == 0:
                print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    (i + iters), 100. * (i + iters)/ (iteration * pretrain_epochs), loss.item()))
        iters += len(source1_loader)
        
    return model

def train(config):
    inner_args = utils.config_inner_args(config.get('inner_args'))
    if config.get('load'):
        ckpt = torch.load(config['load'])
        config['encoder'] = ckpt['encoder']
        config['encoder_args'] = ckpt['encoder_args']
        config['classifier'] = ckpt['classifier']
        config['classifier_args'] = ckpt['classifier_args']
        model = models.load(ckpt,
                            load_clf=(not inner_args['reset_classifier']))
        optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
        start_epoch = ckpt['training']['epoch'] + 1
        max_va = ckpt['training']['max_va']
    else:
        config['encoder_args'] = config.get('encoder_args') or dict()
        config['classifier_args'] = config.get('classifier_args') or dict()
        config['encoder_args']['bn_args']['n_episode'] = config['train'][
            'n_episode']
        config['classifier_args']['n_way'] = config['train']['n_way']
        model = models.make(config['encoder'], config['encoder_args'],
                            config['classifier'], config['classifier_args'])
        optimizer, lr_scheduler = optimizers.make(config['optimizer'],
                                                  model.parameters(),
                                                  **config['optimizer_args'])
    if cuda:
        model.cuda()
    
    start_epoch = 1
    
    if config.get('efficient'):
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    ckpt_name = config['encoder']
    ckpt_name += '_' + config['dataset'].replace('meta-', '')

    ckpt_path = os.path.join('./save', ckpt_name)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))

    model = pretrain(model, optimizer, source1_name)
    ######## Incremental Learning
    for original_source_name in sources:
        if dataset == "office31":
            source_name = original_source_name + "/images/"
            target_name = original_target_name + "/images/"
        else:
            source_name, target_name = original_source_name, original_target_name
        source1_test_loader = data_loader.load_testing(root_path, source_name, batch_size, kwargs)
        target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)
        correct = 0
        for epoch in range(start_epoch, config['epoch'] + 1):
            
            ############### Data
            # 更新hnsw
            index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
            index.set_ef(int(k * 1.2))

            start = time.time()
            source_nums, src_qry_label = test_hnsw(model, source1_test_loader, original_source_name, loader_type=1)
            tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label = test_hnsw(model, target_test_loader, original_target_name, loader_type=0, idx_init=source_nums, k=10)
            end = time.time()
            print("HNSW:", end - start)

            model.train()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            np.random.seed(epoch)
            
            # meta-training dataloader
            tgt_idx, src_indices = index_generate_sim(tgt_idx=tgt_spt_idx, src_idx=tgt_spt_src, src_prob=tgt_spt_src_prob)
            pair1_target1_loader = data_loader.load_training_index(root_path, target_name, batch_size, tgt_idx, kwargs)
            pair1_source1_loader = data_loader.load_training_index(root_path, source_name, batch_size, src_indices, kwargs)
            
            src_qry_idx1, src_qry_idx2 = index_generate_diff(src_indices, src_qry_label[src_indices], shuffle=False)
            pair2_source1_loader2 = data_loader.load_training_index(root_path, source_name, batch_size, src_qry_idx2, kwargs)

            cls_source1_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)

            # meta-testing dataloader
            tgt_qry_idx1, tgt_qry_idx2 = index_generate_diff(tgt_qry_idx, tgt_qry_label, shuffle=True)
            pair_3_target_loader1 = data_loader.load_training_index(root_path, target_name, batch_size, tgt_qry_idx1, kwargs)
            pair_3_target_loader2 = data_loader.load_training_index(root_path, target_name, batch_size, tgt_qry_idx2, kwargs)    


            pair1_iter_a = iter(pair1_target1_loader)
            pair1_iter_b = iter(pair1_source1_loader)

            pair2_iter_b = iter(pair2_source1_loader2)

            trn_cls = iter(cls_source1_loader)

            pair3_iter_a = iter(pair_3_target_loader1)
            pair3_iter_b = iter(pair_3_target_loader2) 

            
            ############### Meta-Trainging
            for data in tqdm(pair1_target1_loader, desc='meta-train', leave=False):
                pair1_tgt_data, pair1_tgt_label = data
                pair1_src_data, pair1_src_label = pair1_iter_b.next()
                trn_pair1 = [pair1_tgt_data, pair1_tgt_label, pair1_src_data, pair1_src_label]
                
                pair2_src_data2, pair2_src_label2 = pair2_iter_b.next()
                trn_pair2 = [pair2_src_data2, pair2_src_label2]
                
                src_trn_data, src_trn_label = trn_cls.next()
                trn_group = [src_trn_data, src_trn_label]
                
                pair3_tgt_data1, pair3_tgt_label1 = pair3_iter_a.next()
                pair3_tgt_data2, pair3_tgt_label2 = pair3_iter_b.next()
                tst_pair3 = [pair3_tgt_data1, pair3_tgt_label1, pair3_tgt_data2, pair3_tgt_label2]
                
                loss = model(trn_pair1,
                               trn_pair2,
                               trn_group,
                               tst_pair3,
                               inner_args,
                               meta_train=True)

                optimizer.zero_grad()
                loss.backward()
                for param in optimizer.param_groups[0]['params']:
                    nn.utils.clip_grad_value_(param, 10)
                optimizer.step()
            

def test_hnsw(model, data_loader, source_name, loader_type=1, idx_init=0, k=10):
    """
    loader_type: 1表示源域，0表示目标域
    """
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    tgt_spt_src = np.empty(shape=(0, k))
    tgt_spt_src_prob = np.empty(shape=(0, k), dtype=np.float32)
    tgt_spt_idx = []
    tgt_qry_idx = []
    tgt_qry_label = []
    src_qry_label = np.empty(shape=(0))
    idx = idx_init
    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data), Variable(target)
            pred, feat = model(None,
                   None,
                   [data, target],
                   None,
                   None,
                   meta_train=False, mark=2)

            pred = torch.nn.functional.softmax(pred, dim=1)
            
            feat = feat.cpu().numpy()
            if loader_type == 0:
                tgt_hard = torch.where(pred.data.max(1)[0] < hard_threshold)[0].cpu().numpy()
                tgt_simple = torch.where(pred.data.max(1)[0] >= hard_threshold)[0].cpu().numpy()
                tgt_spt_idx = np.concatenate((tgt_spt_idx, [x + idx for x in tgt_hard]), axis=0)
                tgt_qry_idx = np.concatenate((tgt_qry_idx, [x + idx for x in tgt_simple]), axis=0)
                tgt_pred = pred.data.max(1)[1].cpu().numpy()
                tgt_qry_label = np.concatenate((tgt_qry_label, tgt_pred[tgt_simple]), axis=0)
                for spt_sample in tgt_hard:
                    sample_label = spt_sample + idx
                    index.add_items(feat[spt_sample], sample_label)
                    labels, distances = index.knn_query(feat[spt_sample], k=k)
                    tgt_spt_src = np.concatenate((tgt_spt_src, labels), axis=0)
                    tgt_spt_src_prob = np.concatenate((tgt_spt_src_prob, softmax(distances)), axis=0)
                    index.mark_deleted(sample_label)
            else:
                index.add_items(feat, [x + idx for x in range(len(data))])
                src_qry_label = np.concatenate([src_qry_label, target], axis=0)
            
            pred = pred.data.max(1)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            idx += len(data)

        print(source_name, 'Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))
    if loader_type == 1:
        return idx, src_qry_label
    else:
        return tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label
            
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark = 0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred1 + pred2) / 2
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
    return correct

def index_generate_sim(tgt_idx, src_idx, src_prob):
    src_indices = [np.random.choice(x, size=1, p=src_prob[i]) for i,x in enumerate(src_idx)]
    idx_pair = list(zip(tgt_idx, src_indices))
    np.random.shuffle(idx_pair)
    tgt_idx, src_indices = zip(*idx_pair)
    tgt_idx = np.array(tgt_idx, dtype=np.int64).flatten()
    src_indices = np.array(src_indices, dtype=np.int64).flatten()
    return tgt_idx, src_indices


def index_generate_diff(tgt_qry_idx, tgt_qry_label, shuffle):
    """
    优化：类别概率，样本概率
    """
    d = defaultdict(list)
    for s, l in zip(tgt_qry_idx, tgt_qry_label):
        d[l].append(s)
    category_list = list(d.keys())
    tgt_qry_idx_aux = [np.random.choice(d[random_chice_the_other(tgt_qry_label[i],category_list)], size=1) for i in range(len(tgt_qry_idx))]
    if shuffle:
        return zip(*np.random.shuffle(tgt_qry_idx, tgt_qry_idx_aux))
    else:
        return tgt_qry_idx, tgt_qry_idx_aux
    
def random_chice_the_other(x, categorys):
    categorys = set(categorys)
    categorys.discard(x)
    return np.random.choice(list(categorys), size=1)
    

    
def softmax(x):
    """ softmax function """
    
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    x /= x.sum()
    
    return x


if __name__ == '__main__':
    config = yaml.load(open("configs/train.yaml", 'r'), Loader=yaml.FullLoader)
    train(config)
