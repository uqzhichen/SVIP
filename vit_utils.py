import pprint
import json
from os import path as osp
from pathlib import Path
from collections import namedtuple
import os, sys, time, torch
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# self-packages
import copy
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=4)

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

def get_attr_group(name):
    if "CUB" in name:
        attr_group = {
            1: [i for i in range(0, 9)],
            2: [i for i in range(9, 24)],
            3: [i for i in range(24, 39)],
            4: [i for i in range(39, 54)],
            5: [i for i in range(54, 58)],
            6: [i for i in range(58, 73)],
            7: [i for i in range(73, 79)],
            8: [i for i in range(79, 94)],
            9: [i for i in range(94, 105)],
            10: [i for i in range(105, 120)],
            11: [i for i in range(120, 135)],
            12: [i for i in range(135, 149)],
            13: [i for i in range(149, 152)],
            14: [i for i in range(152, 167)],
            15: [i for i in range(167, 182)],
            16: [i for i in range(182, 197)],
            17: [i for i in range(197, 212)],
            18: [i for i in range(212, 217)],
            19: [i for i in range(217, 222)],
            20: [i for i in range(222, 236)],
            21: [i for i in range(236, 240)],
            22: [i for i in range(240, 244)],
            23: [i for i in range(244, 248)],
            24: [i for i in range(248, 263)],
            25: [i for i in range(263, 278)],
            26: [i for i in range(278, 293)],
            27: [i for i in range(293, 308)],
            28: [i for i in range(308, 312)],
        }

    elif "AwA" in name or "AWA" in name:
        attr_group = {
            1: [i for i in range(0, 8)],
            2: [i for i in range(8, 14)],
            3: [i for i in range(14, 18)],
            4: [i for i in range(18, 34)] + [44, 45],
            5: [i for i in range(34, 44)],
            6: [i for i in range(46, 51)],
            7: [i for i in range(51, 63)],
            8: [i for i in range(63, 78)],
            9: [i for i in range(78, 85)],
        }

    elif "SUN" in name:
        attr_group = {
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            2: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73] + [80, 99],
            3: [74, 75, 76, 77, 78, 79, 81, 82, 83, 84, ],
            4: [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98] + [100, 101]
        }
    else:
        attr_group = {}

    return attr_group

def count_parameters_in_MB(model):
    if isinstance(model, nn.Module):
        return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    else:
        return np.sum(np.prod(v.size()) for v in model) / 1e6


class Logger(object):

    def __init__(self, log_dir, seed):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path("{:}".format(str(log_dir)))
        if not self.log_dir.exists(): os.makedirs(str(self.log_dir))

        self.log_file = '{:}/log-{:}-date-{:}.txt'.format(self.log_dir, seed, time_for_file())
        self.file_writer = open(self.log_file, 'w')

    def checkpoint(self, name):
        return self.log_dir / name

    def print(self, string, fprint=True, is_pp=False):
        if is_pp: pp.pprint (string)
        else:     print(string)
        if fprint:
            self.file_writer.write('{:}\n'.format(string))
            self.file_writer.flush()

    def close(self):
        self.file_writer.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def obtain_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # bs*k
        pred = pred.t()  # t: transpose, k*bs
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 1*bs --> k*bs

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def obtain_per_class_accuracy(predictions, xtargets):
    top1 = torch.argmax(predictions, dim=1)
    cls2accs = []
    for cls in sorted(list(set(xtargets.tolist()))):
        selects  = xtargets == cls
        accuracy = (top1[selects] == xtargets[selects]).float().mean() * 100
        cls2accs.append( accuracy.item() )
    return sum(cls2accs) / len(cls2accs)


def convert_secs2time(epoch_time, string=True):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    if string:
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        return need_time
    else:
        return need_hour, need_mins, need_secs

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d-%X'
    string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string

def time_for_file():
    ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
    string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string

def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

support_types = ('str', 'int', 'bool', 'float', 'none')


def convert_param(original_lists):
    assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list: value = [value]
    outs = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            assert x == 'None', 'for none type, the value must be None instead of {:}'.format(x)
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    if not is_list: outs = outs[0]
    return outs


def load_configure(path, extra, logger):
    if path is not None:
        path = str(path)
        if hasattr(logger, 'print'): logger.print(path)
        assert os.path.exists(path), 'Can not find {:}'.format(path)
        # Reading data back
        with open(path, 'r') as f:
          data = json.load(f)
        f.close()
        content = { k: convert_param(v) for k,v in data.items()}
    else:
        content = {}
    assert extra is None or isinstance(extra, dict), 'invalid type of extra : {:}'.format(extra)
    if isinstance(extra, dict): content = {**content, **extra}
    Arguments = namedtuple('Configure', ' '.join(content.keys()))
    content   = Arguments(**content)
    if hasattr(logger, 'print'): logger.print('{:}'.format(content))
    return content


def Entropy(x):
    bs = x.size(0)
    epsilon = 1e-10
    entropy = -F.softmax(x, dim=1) * F.log_softmax(x + epsilon, dim=1)
    entropy = torch.mean(entropy, dim=1)
    return entropy

# class HLoss(nn.Module):
#     def __init__(self):
#         super(HLoss, self).__init__()
#
#     def forward(self, x):
#         b = F.softmax(x, dim=0) * F.log_softmax(x, dim=0)
#         b = -1.0 * b.sum()
#         return b

def configure2str(config, xpath=None):
    if not isinstance(config, dict):
        config = config._asdict()
    def cstring(x):
        return "\"{:}\"".format(x)
    def gtype(x):
        if isinstance(x, list): x = x[0]
        if isinstance(x, str)  : return 'str'
        elif isinstance(x, bool) : return 'bool'
        elif isinstance(x, int): return 'int'
        elif isinstance(x, float): return 'float'
        elif x is None           : return 'none'
        else: raise ValueError('invalid : {:}'.format(x))
    def cvalue(x, xtype):
        if isinstance(x, list): is_list = True
        else:
            is_list, x = False, [x]
        temps = []
        for temp in x:
            if xtype == 'bool'  : temp = cstring(int(temp))
            elif xtype == 'none': temp = cstring('None')
            else                : temp = cstring(temp)
            temps.append( temp )
        if is_list:
            return "[{:}]".format( ', '.join( temps ) )
        else:
            return temps[0]

    xstrings = []
    for key, value in config.items():
        xtype  = gtype(value)
        string = '  {:20s} : [{:8s}, {:}]'.format(cstring(key), cstring(xtype), cvalue(value, xtype))
        xstrings.append(string)
    Fstring = '{\n' + ',\n'.join(xstrings) + '\n}'
    if xpath is not None:
        parent = Path(xpath).resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        if osp.isfile(xpath): os.remove(xpath)
        with open(xpath, "w") as text_file:
            text_file.write('{:}'.format(Fstring))
    return Fstring


def dict2configure(xdict, logger):
    assert isinstance(xdict, dict), 'invalid type : {:}'.format( type(xdict) )
    Arguments = namedtuple('Configure', ' '.join(xdict.keys()))
    content   = Arguments(**xdict)
    if hasattr(logger, 'print'): logger.print('{:}'.format(content))
    return content


def evaluate_dual(loader, features, transformer, scale):
    losses, all_predictions, all_labels = AverageMeter(), [], []
    transformer.eval()
    probs = []
    # head_1 = nn.Linear(15000, 312, bias=False).cuda()
    # head_1.weight.data = torch.randn(15000, 312).cuda()
    # features = (head_1(features))
    with torch.no_grad():
        # head_1 = nn.Linear(312, 15000).cuda()

        for batch_idx, (image_features, target) in enumerate(loader):
            # att_head, query, _ , _, _ = transformer(image_features.cuda(), retrieved_samples=None, inference=True)
            att_head = transformer(image_features.cuda())
            # att_head, _, _  = transformer(image_features.cuda())
            # att_head = (head_1(att_head))
            # gs_feat_norm = torch.norm(att_head, p=2, dim=1).unsqueeze(1).expand_as(att_head)
            # gs_feat_normalized = att_head.div(gs_feat_norm + 1e-5)
            # temp_norm = torch.norm(features, p=2, dim=1).unsqueeze(1).expand_as(features)
            # features = features.div(temp_norm + 1e-5)
            cos_dist = torch.einsum('bd,nd->bn', att_head, features)
            sim = cos_dist*5
            # sim = torch.matmul(att_pred, features.T)*scale
            sim = F.softmax(sim, dim=1)
            predict_labels = torch.argmax(sim, dim=1).cpu()
            all_predictions.append(predict_labels)
            all_labels.append(target)
            probs.append(sim)
    predictions = torch.cat(all_predictions, dim=0)
    labels      = torch.cat(all_labels, dim=0)
    probs = torch.cat(probs, dim=0)
    all_classes = sorted(list(set(labels.tolist())))
    acc_per_classes = []
    for idx, cls in enumerate(all_classes):
        assert idx <= cls, 'invalid all-classes : {:}'.format(all_classes)
        #print ('dataset : {:}'.format(loader.dataset))
        indexes = labels == cls
        xpreds, xlabels = predictions[indexes], labels[indexes]
        acc_per_classes.append( (xpreds==xlabels).float().mean().item() )
    acc_per_class = float(np.mean(acc_per_classes))
    return ['{:3.1f}'.format(x*100) for x in acc_per_classes], acc_per_class * 100, probs


def evaluate_all_dual(epoch_str, test_unseen_loader, test_seen_loader, features, transformer,  info, best_accs,
                      best_stacked_accs, logger, scale):
    train_classes, unseen_classes = info['train_classes'], info['unseen_classes']
    logger.print('Evaluate [{:}]'.format(epoch_str))

    # calculate zero shot setting
    test_unseen_loader.dataset.set_return_label_mode('new')
    target_semantics  = features[unseen_classes, :]
    _, test_per_cls_acc, _ = evaluate_dual(test_unseen_loader, target_semantics, transformer, scale)
    if best_accs['zs'] < test_per_cls_acc:
        best_accs['zs'] = test_per_cls_acc
    logger.print('Test {:} [zero-zero-zero-zero-shot----] {:} done, '
                 'per-class-acc={:5.2f}% (TTEST-best={:5.2f}%).'.format(time_string(),
                                                        epoch_str, test_per_cls_acc, best_accs['zs']))
    # calculate generalized zero-shot setting
    test_unseen_loader.dataset.set_return_label_mode('original')
    test_unsn_accs, test_per_cls_acc_unseen, probs_unseen  = evaluate_dual(test_unseen_loader, features, transformer, scale)

    if best_accs['gzs-unseen'] < test_per_cls_acc_unseen:
        best_accs['gzs-unseen'] = test_per_cls_acc_unseen
    logger.print('Test {:} [generalized-zero-shot-unseen] {:} done, per-class-acc={:5.2f}% (TUNSN-best={:5.2f}%).'.format(
                    time_string(), epoch_str, test_per_cls_acc_unseen, best_accs['gzs-unseen']))

    #logger.print('Test {:} [generalized-zero-shot-unseen] {:} ::: {:}.'.format(time_string(), epoch_str, test_unsn_accs))
    # for test data with seen classes
    test_seen_loader.dataset.set_return_label_mode('original')
    test_seen_accs, test_per_cls_acc_seen, probs_seen = evaluate_dual(test_seen_loader, features,  transformer, scale)

    if best_accs['gzs-seen'] < test_per_cls_acc_seen:
        best_accs['gzs-seen'] = test_per_cls_acc_seen
    logger.print('Test {:} [generalized-zero-shot---seen] {:} done,per-class-acc={:5.2f}% (TSEEN-best={:5.2f}%).'.format(
                    time_string(), epoch_str, test_per_cls_acc_seen, best_accs['gzs-seen']))
    #logger.print('Test {:} [generalized-zero-shot---seen] {:} ::: {:}.'.format(time_string(), epoch_str, test_seen_accs))
    harmonic_mean        = (2 * test_per_cls_acc_seen * test_per_cls_acc_unseen) / (test_per_cls_acc_seen + test_per_cls_acc_unseen + 1e-8)

    if best_accs['gzs-H'] < harmonic_mean:
        best_accs['gzs-H'] = harmonic_mean
        best_accs['best-info'] = '[{:}] seen={:5.2f}% unseen={:5.2f}%, H={:5.2f}%'.format(epoch_str, test_per_cls_acc_seen, test_per_cls_acc_unseen, harmonic_mean)
    logger.print('Test [generalized-zero-shot-h-mean] {:} H={:.3f}% (HH-best={:.3f}%). ||| Best comes from {:}'.format(epoch_str, harmonic_mean, best_accs['gzs-H'], best_accs['best-info']))

    acc_S_T_list, acc_U_T_list, H_list = list(), list(), list()
    for e in np.arange(-1, 1, 0.01):
        tmp_seen_sim = copy.deepcopy(probs_seen)
        tmp_seen_sim[:, unseen_classes] += e
        pred_lbl = torch.argmax(tmp_seen_sim, axis=1)
        acc_S_T_list.append((pred_lbl == torch.tensor(test_seen_loader.dataset.labels).cuda()).float().mean())

        tmp_unseen_sim = copy.deepcopy(probs_unseen)
        tmp_unseen_sim[:, unseen_classes] += e
        pred_lbl = torch.argmax(tmp_unseen_sim, axis=1)
        acc_U_T_list.append((pred_lbl ==torch.tensor(test_unseen_loader.dataset.labels).cuda()).float().mean())

    for i, j in zip(acc_S_T_list, acc_U_T_list):
        H = 100* 2 * i * j / (i + j)
        H_list.append(H)
    max_H = max(H_list)
    max_idx = H_list.index(max_H)
    max_U = acc_U_T_list[max_idx]*100
    max_S = acc_S_T_list[max_idx]*100
    better_model = False
    if best_stacked_accs['gzs-unseen'] < max_U:
        best_stacked_accs['gzs-unseen'] = max_U
    if best_stacked_accs['gzs-seen'] < max_S:
        best_stacked_accs['gzs-seen'] = max_S
    if best_stacked_accs['gzs-H'] < max_H:
        best_stacked_accs['gzs-H'] = max_H
        best_stacked_accs['best-info'] = '[{:}] seen={:5.2f}% unseen={:5.2f}%, H={:5.2f}%'.format(epoch_str, max_S, max_U, max_H)
        better_model = True
    logger.print('Test {:} [generalized-zero-shot-unseen] {:} done, per-class-acc={:5.2f}% (TUNSN-best={:5.2f}%).'.format(
                    time_string(), epoch_str, max_U, best_stacked_accs['gzs-unseen']))
    logger.print('Test Stacked {:} [generalized-zero-shot---seen] {:} done,per-class-acc={:5.2f}% (TSEEN-best={:5.2f}%).'.format(
                    time_string(), epoch_str, max_S, best_stacked_accs['gzs-seen']))
    logger.print('Test Stacked [generalized-zero-shot-h-mean] {:} H={:.3f}% (HH-best={:.3f}%). ||| '
                 'Best comes from {:}'.format(epoch_str, max_H, best_stacked_accs['gzs-H'], best_stacked_accs['best-info']))


    return better_model