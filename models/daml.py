from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d
import numpy as np
import pdb

def make(enc_name, enc_args, clf_name, clf_args):
    """
  Initializes a random meta model.
  Args:
    enc_name (str): name of the encoder (e.g., 'resnet12').
    enc_args (dict): arguments for the encoder.
    clf_name (str): name of the classifier (e.g., 'meta-nn').
    clf_args (dict): arguments for the classifier.
  Returns:
    model (MAML): a meta classifier with a random encoder.
  """
    enc = encoders.make(enc_name, **enc_args)
    clf_args['in_dim'] = enc.get_out_dim()
    clf = classifiers.make(clf_name, **clf_args)
    model = MAML(enc, clf)
    return model


def load(ckpt, load_clf=False, clf_name=None, clf_args=None):
    """
  Initializes a meta model with a pre-trained encoder.
  Args:
    ckpt (dict): a checkpoint from which a pre-trained encoder is restored.
    load_clf (bool, optional): if True, loads a pre-trained classifier.
      Default: False (in which case the classifier is randomly initialized)
    clf_name (str, optional): name of the classifier (e.g., 'meta-nn')
    clf_args (dict, optional): arguments for the classifier.
    (The last two arguments are ignored if load_clf=True.)
  Returns:
    model (MAML): a meta model with a pre-trained encoder.
  """
    enc = encoders.load(ckpt)
    if load_clf:
        clf = classifiers.load(ckpt)
    else:
        if clf_name is None and clf_args is None:
            clf = classifiers.make(ckpt['classifier'],
                                   **ckpt['classifier_args'])
        else:
            clf_args['in_dim'] = enc.get_out_dim()
            clf = classifiers.make(clf_name, **clf_args)
    model = MAML(enc, clf)
    return model


class MAML(Module):
    def __init__(self, encoder, classifier):
        super(MAML, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def reset_classifier(self):
        self.classifier.reset_parameters()
    
    def _outer_forward(self, x, params, episode):
        """ Forward pass for the inner loop. """
        feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
        return feat
    
    def _inner_forward(self, x, params, episode):
        """ Forward pass for the inner loop. """
        feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
        logits = self.classifier(feat, get_child_dict(params, 'classifier'))
        return logits, feat

    def _inner_iter(self, trn_pair1, trn_pair2, trn_group, params, mom_buffer, episode, inner_args,
                    detach):
        """ 
        Performs one inner-loop iteration of MAML including the forward and 
        backward passes and the parameter update.
        Args:
          x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
          y (int tensor, [n_way * n_shot]): per-episode support set labels.
          params (dict): the model parameters BEFORE the update.
          mom_buffer (dict): the momentum buffer BEFORE the update.
          episode (int): the current episode index.
          inner_args (dict): inner-loop optimization hyperparameters.
          detach (bool): if True, detachs the graph for the current iteration.
        Returns:
          updated_params (dict): the model parameters AFTER the update.
          mom_buffer (dict): the momentum buffer AFTER the update.
        """
        with torch.enable_grad():
            # forward pass
            src_trn_data, src_trn_label = trn_group
            src_trn_data, src_trn_label = src_trn_data.cuda(), src_trn_label.cuda()
            logits, _ = self._inner_forward(src_trn_data, params, episode)
            loss = F.cross_entropy(logits, src_trn_label)
            
            pair1_tgt_data, i, pair1_src_data, j = trn_pair1
            pair1_tgt_data, pair1_src_data = pair1_tgt_data.cuda(), pair1_src_data.cuda()
            pair1_feat_tgt = self._outer_forward(pair1_tgt_data, params, episode)
            pair1_feat_src = self._outer_forward(pair1_src_data, params, episode)
            
            feat_diff = pair1_feat_tgt - pair1_feat_src
            loss += torch.mean(torch.norm(feat_diff, p=2, dim=1))

            pair2_src_data2, q = trn_pair2
            pair2_src_data2 = pair2_src_data2.cuda()
            pair2_feat_src = self._outer_forward(pair2_src_data2, params, episode)
            loss += self._diff_loss(pair1_feat_src, pair2_feat_src)
            
            # backward pass
            grads = autograd.grad(
                loss,
                params.values(),
                create_graph=(not detach and not inner_args['first_order']),
                only_inputs=True,
                allow_unused=True)
            # parameter update
            updated_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                if grad is None:
                    updated_param = param
                else:
                    if inner_args['weight_decay'] > 0:
                        grad = grad + inner_args['weight_decay'] * param
                    if inner_args['momentum'] > 0:
                        grad = grad + inner_args['momentum'] * mom_buffer[name]
                        mom_buffer[name] = grad
                    if 'encoder' in name:
                        lr = inner_args['encoder_lr']
                    elif 'classifier' in name:
                        lr = inner_args['classifier_lr']
                    else:
                        raise ValueError('invalid parameter name')
                    updated_param = param - lr * grad
                if detach:
                    updated_param = updated_param.detach().requires_grad_(True)
                updated_params[name] = updated_param

        return updated_params, mom_buffer

    def _adapt(self, trn_pair1, trn_pair2, trn_group, params, inner_args, meta_train, episode):
        """
    Performs inner-loop adaptation in MAML.
    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """

        # Initializes a dictionary of momentum buffer for gradient descent in the
        # inner loop. It has the same set of keys as the parameter dictionary.
        mom_buffer = OrderedDict()
        if inner_args['momentum'] > 0:
            for name, param in params.items():
                mom_buffer[name] = torch.zeros_like(param)
        params_keys = tuple(params.keys())
        mom_buffer_keys = tuple(mom_buffer.keys())

        for m in self.modules():
            if isinstance(m, BatchNorm2d) and m.is_episodic():
                m.reset_episodic_running_stats(episode)

        def _inner_iter_cp(episode, *state):
            """ 
      Performs one inner-loop iteration when checkpointing is enabled. 
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
            params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
            mom_buffer = OrderedDict(
                zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

            detach = not torch.is_grad_enabled(
            )  # detach graph in the first pass
            self.is_first_pass(detach)
            params, mom_buffer = self._inner_iter(trn_pair1, trn_pair2, trn_group, params, mom_buffer,
                                                  int(episode), inner_args,
                                                  detach)
            state = tuple(
                t if t.requires_grad else t.clone().requires_grad_(True)
                for t in tuple(params.values()) + tuple(mom_buffer.values()))
            return state

        for step in range(inner_args['n_step']):
            if self.efficient:  # checkpointing
                state = tuple(params.values()) + tuple(mom_buffer.values())
                state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode),
                                      *state)
                params = OrderedDict(zip(params_keys,
                                         state[:len(params_keys)]))
                mom_buffer = OrderedDict(
                    zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
            else:
                params, mom_buffer = self._inner_iter(trn_pair1, trn_pair2, trn_group, params, mom_buffer,
                                                      episode, inner_args,
                                                      not meta_train)

        return params

    def forward(self, trn_pair1, trn_pair2, trn_group, tst_pair3, inner_args, meta_train, mark=1):
        """
        Args:
          x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
          x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
            (T: transforms, C: channels, H: height, W: width)
          y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
          inner_args (dict, optional): inner-loop hyperparameters.
          meta_train (bool): if True, the model is in meta-training.

        Returns:
          logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
        """
        
        
        
        assert self.encoder is not None
        assert self.classifier is not None

        self.train()
        if mark == 0:
            src_trn_data, src_trn_label = trn_group
            src_trn_data, src_trn_label = src_trn_data.cuda(), src_trn_label.cuda()
            logits, _ = self._inner_forward(src_trn_data, OrderedDict(self.named_parameters()), 0)
            loss = F.cross_entropy(logits, src_trn_label)
        elif mark == 1:
            # a dictionary of parameters that will be updated in the inner loop
            params = OrderedDict(self.named_parameters())
            for name in list(params.keys()):
                if not params[name].requires_grad or \
                  any(s in name for s in inner_args['frozen'] + ['temp']):
                    params.pop(name)
            # inner-loop training
            pair3_tgt_data1, m, pair3_tgt_data2, n = tst_pair3
            if not meta_train:
                for m in self.modules():
                    if isinstance(m, BatchNorm2d) and not m.is_episodic():
                        m.eval()
            updated_params = self._adapt(trn_pair1, trn_pair2, trn_group, params,
                                         inner_args, meta_train, 0)
            # inner-loop validation
            with torch.set_grad_enabled(meta_train):
                self.eval()
                pair3_tgt_data1, pair3_tgt_data2 = pair3_tgt_data1.cuda(), pair3_tgt_data2.cuda()
                feat1 = self._outer_forward(pair3_tgt_data1, updated_params, 0)
                feat2 = self._outer_forward(pair3_tgt_data2, updated_params, 0)
                loss = self._diff_loss(feat1, feat2)

            self.train(meta_train)
        
        if mark == 2:
            src_trn_data, src_trn_label = trn_group
            src_trn_data, src_trn_label = src_trn_data.cuda(), src_trn_label.cuda()
            logits, feat = self._inner_forward(src_trn_data, OrderedDict(self.named_parameters()), 0)
            return logits, feat
        return loss

    def _diff_loss(self, data1, data2):
        try:
            data1_norm = torch.norm(data1, p=2, dim=0)
            data2_norm = torch.norm(data2, p=2, dim=0)
            data1 = data1 / data1_norm
            data2 = data2 / data2_norm
            channel_sim = torch.matmul(data1.transpose(0,1), data2) / np.sqrt(data1.size(0))
        except:
            pdbl.set_trace()
        return torch.mean(torch.abs(torch.sum(channel_sim, dim=1)))
