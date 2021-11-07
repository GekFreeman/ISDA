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
from tqdm import tqdm
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Training settings
pretrain_epochs = 5
batch_size = 16
iteration = 10000
lr = [0.01, 0.01]
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
    
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}


def pretrain(model, optimizer, source_name, target_name, lr_scheduler, log=False, epochs=10, writer=None):
    if dataset == "office31":
        new_source_name = source_name + '/images/'
        new_target_name = target_name + '/images/'
    source1_loader = data_loader.load_training(root_path, new_source_name, batch_size, kwargs)
    source1_test_loader = data_loader.load_testing(root_path, new_source_name, batch_size, kwargs)
    target_test_loader = data_loader.load_testing(root_path, new_target_name, batch_size, kwargs)
    source1_iter = iter(source1_loader)
    iters = 0
    iteration = len(source1_loader)
    for ep in range(epochs):
        for i, data in enumerate(source1_loader):
            optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i + iters) / (iteration * epochs)), 0.75)
            
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
            if log:
                writer.add_scalar(f'pretrain_{source_name}_loss', loss.item(), i+iters)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i+iters)
        
        iters += len(source1_loader)
        src_acc = test_acc(model, source1_test_loader, source_name, source_type="Source")
        tgt_acc = test_acc(model, target_test_loader, target_name, source_type="Target")
        if log:
            writer.add_scalar(f'pretrain_{source_name}_acc', src_acc, ep)
            writer.add_scalar(f'pretrain_{target_name}_acc', tgt_acc, ep)
#         if lr_scheduler is not None:
#             lr_scheduler.step()
        writer.flush()
        
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

    model = pretrain(model, optimizer, sources[0], original_target_name, lr_scheduler, epochs=config['pre_train'], log=True, writer=writer)
    assert 1==0
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
        
        optimizer, lr_scheduler = optimizers.make(config['optimizer'],
                                    model.parameters(),
                                    **config['optimizer_args'])
        
        ####meta-learning
        for epoch in range(start_epoch, config['meta_epochs'] + 1):
            
            ############### Data
            # 更新hnsw
            index = hnswlib.Index(space='l2', dim=dim) # dim 向量维度
            index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
            index.set_ef(int(k * 1.2))

            start = time.time()
            source_nums, src_qry_label, acc_src = test_hnsw(model, source1_test_loader, original_source_name, loader_type=1, index=index)
            tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label, acc_tgt = test_hnsw(model, target_test_loader, original_target_name, loader_type=0, idx_init=source_nums, k=10, index=index)
            end = time.time()
            print("HNSW:", end - start)
            
            writer.add_scalar(f'source_{source_name}_acc', acc_src, epoch)
            writer.add_scalar(f'target_{target_name}_acc', acc_tgt, epoch)
            
            model.train()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            np.random.seed(epoch)
            
            # meta-training dataloader
            tgt_idx, src_indices = index_generate_sim(tgt_idx=tgt_spt_idx, src_idx=tgt_spt_src, src_prob=tgt_spt_src_prob)
            pair1_target1_loader = data_loader.load_training_index(root_path, target_name, batch_size, tgt_idx, kwargs)
            pair1_source1_loader = data_loader.load_training_index(root_path, source_name, batch_size, src_indices, kwargs)

            src_qry_idx1, _, src_qry_idx2, _ = index_generate_diff(src_indices, src_qry_label[src_indices], shuffle=False)
            pair2_source1_loader2 = data_loader.load_training_index(root_path, source_name, batch_size, src_qry_idx2, kwargs)

            cls_source1_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)

            # meta-testing dataloader
            tgt_qry_idx1, tgt_qry_label1, tgt_qry_idx2, tgt_qry_label2 = index_generate_diff(tgt_qry_idx, tgt_qry_label, shuffle=True)
            pair_3_target_loader1 = data_loader.load_training_index(root_path, target_name, batch_size, tgt_qry_idx1, kwargs, target=True, pseudo=tgt_qry_label1)
            pair_3_target_loader2 = data_loader.load_training_index(root_path, target_name, batch_size, tgt_qry_idx2, kwargs, target=True, pseudo=tgt_qry_label2)    


            pair1_iter_a = iter(pair1_target1_loader)
            pair1_iter_b = iter(pair1_source1_loader)

            pair2_iter_b = iter(pair2_source1_loader2)

            trn_cls = iter(cls_source1_loader)

            pair3_iter_a = iter(pair_3_target_loader1)
            pair3_iter_b = iter(pair_3_target_loader2) 
            
            ############### Meta-Trainging
            iters = 0
            for data in tqdm(pair1_target1_loader, desc='meta-train', leave=False):
                pair1_tgt_data, pair1_tgt_label = data
                
                
                pair1_src_data, pair1_src_label,  pair1_iter_b = save_iter(pair1_iter_b, pair1_source1_loader)
                trn_pair1 = [pair1_tgt_data, pair1_tgt_label, pair1_src_data, pair1_src_label]

                pair2_src_data2, pair2_src_label2, pair2_iter_b = save_iter(pair2_iter_b, pair2_source1_loader2)
                trn_pair2 = [pair2_src_data2, pair2_src_label2]
                
                src_trn_data, src_trn_label, trn_cls = save_iter(trn_cls, cls_source1_loader)
                trn_group = [src_trn_data, src_trn_label]
                
                pair3_tgt_data1, pair3_tgt_label1, pair3_iter_a = save_iter(pair3_iter_a, pair_3_target_loader1)
                pair3_tgt_data2, pair3_tgt_label2, pair3_iter_b = save_iter(pair3_iter_b, pair_3_target_loader2)
                tst_pair3 = [pair3_tgt_data1, pair3_tgt_label1, pair3_tgt_data2, pair3_tgt_label2]
                
                loss = model(trn_pair1,
                               trn_pair2,
                               trn_group,
                               tst_pair3,
                               inner_args,
                               meta_train=True)

                writer.add_scalars('loss', {'meta-train': loss.item()}, iters)
                optimizer.zero_grad()
                loss.backward()
                for param in optimizer.param_groups[0]['params']:
                    nn.utils.clip_grad_value_(param, 10)
                optimizer.step()
                writer.flush()
                iters += len(data)
                
            if lr_scheduler is not None:
                lr_scheduler.step()
        
        ####fine-tune
        model = pretrain(model, optimizer, original_source_name, writer=writer, epochs=config['fine_tune'])
                
def test_acc(model, data_loader, source_name, source_type="source"):
    """
    loader_type: 1表示源域，0表示目标域
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data), Variable(target)
            pred, _ = model(None,
                   None,
                   [data, target],
                   None,
                   None,
                   meta_train=False, mark=2)

            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = pred.data.max(1)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        acc = 100. * correct / len(data_loader.dataset)
#         print(source_dtype, ": ", source_name, 'Accuracy: {}/{} ({:.0f}%)\n'.format(
#          correct, len(data_loader.dataset), acc))
    return acc    
            

def test_hnsw(model, data_loader, source_name, loader_type=1, idx_init=0, k=10, index=None):
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
    domain_idx = 0
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
                tgt_spt_idx = np.concatenate((tgt_spt_idx, [x + domain_idx for x in tgt_hard]), axis=0)
                tgt_qry_idx = np.concatenate((tgt_qry_idx, [x + domain_idx for x in tgt_simple]), axis=0)
                tgt_pred = pred.data.max(1)[1].cpu().numpy()
                tgt_qry_label = np.concatenate((tgt_qry_label, tgt_pred[tgt_simple]), axis=0)
                for spt_sample in tgt_hard:
                    sample_label = spt_sample + idx
#                     index.add_items(feat[spt_sample], sample_label)
                    labels, distances = index.knn_query(feat[spt_sample], k=k)
#                     labels = labels.flatten()
#                     distances = distances.flatten();pdb.set_trace()
#                     labels, distances = zip(*[(l, d) for l, d in zip(labels, distances) if l != sample_label]) # 删除搜索节点自身
                    
#                     labels = np.array(labels, dtype=np.int64).reshape(1, k)
#                     distances = np.array(distances, dtype=np.float32).reshape(1, k)
                    
                    tgt_spt_src = np.concatenate((tgt_spt_src, labels), axis=0)
                    tgt_spt_src_prob = np.concatenate((tgt_spt_src_prob, softmax(distances)), axis=0)
#                     index.mark_deleted(sample_label)
            else:
                index.add_items(feat, [x + idx for x in range(len(data))])
                src_qry_label = np.concatenate([src_qry_label, target], axis=0)
            
            pred = pred.data.max(1)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            idx += len(data)
            domain_idx += len(data)
        
        acc = 100. * correct / len(data_loader.dataset)
        print(source_name, 'Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(data_loader.dataset), acc))
    if loader_type == 1:
        return idx, src_qry_label, acc
    else:
        return tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label, acc

def save_iter(loader_iterator, dataloader):
    try:
        data, label = loader_iterator.next()
    except:
        loader_iterator = iter(dataloader)
        data, label = loader_iterator.next()
    return data, label, loader_iterator
        

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
    id2label = dict(zip(tgt_qry_idx, tgt_qry_label))
    category_list = list(d.keys())
    tgt_qry_idx_aux = [np.random.choice(d[random_chice_the_other(tgt_qry_label[i],category_list)], size=1) for i in range(len(tgt_qry_idx))]
    if shuffle:
        idx_pair = list(zip(tgt_qry_idx, tgt_qry_idx_aux))
        np.random.shuffle(idx_pair)
        tgt_qry_idx, tgt_qry_idx_aux = zip(*idx_pair)
    tgt_qry_idx = np.array(tgt_qry_idx, dtype=np.int64).flatten()
    tgt_qry_idx_aux = np.array(tgt_qry_idx_aux, dtype=np.int64).flatten()
    
    tgt_qry_label = np.array([id2label[x] for x in tgt_qry_idx], dtype=np.int64)
    tgt_qry_label_aux = np.array([id2label[x] for x in tgt_qry_idx_aux], dtype=np.int64)
    return tgt_qry_idx, tgt_qry_label, tgt_qry_idx_aux, tgt_qry_label_aux
    
def random_chice_the_other(x, categorys):
    categorys = set(categorys)
    categorys.discard(x)
    return int(np.random.choice(list(categorys), size=1).item())
    

    
def softmax(x):
    """ softmax function """
    
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    x /= x.sum()
    
    return x


if __name__ == '__main__':
    config = yaml.load(open("configs/train.yaml", 'r'), Loader=yaml.FullLoader)
    train(config)
