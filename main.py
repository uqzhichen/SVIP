import glob
import torch.nn
import numpy as np
from pathlib import Path
from copy import deepcopy
from dataset import ZSLDataset
import torch.nn.functional as F
import os, time, random, argparse
from torch.utils.data import DataLoader
from models.vit_model import VisionTransformer
from vit_utils import Logger, time_string, convert_secs2time, \
    AverageMeter, evaluate_all_dual, get_attr_group


parser = argparse.ArgumentParser(description='SVIP', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Experiment parameters
parser.add_argument('--exp_name', type=str, default="SVIP", help='Experiment name.')

parser.add_argument('--log_dir', type=str, help='Save dir.')

parser.add_argument('--data_root', type=str, default="info-files/", help='dataset root')
parser.add_argument('--dataset', type=str, default="CUB", help='dataset name, i.e., CUB, AWA2, SUN')

# Optimization options
parser.add_argument('--bs', type=int, default=64, help='The number of classes in each episode.')
parser.add_argument('--epochs', type=int, default=30, help='The number of training epochs.')
parser.add_argument('--manual_seed', type=int, default=26961, help='The manual seed.')
parser.add_argument("--attribute_dim", type=int, default=312, help="Dimensionality of the latent space")

parser.add_argument('--pre_lr', type=float, default=1e-3, help='The learning rate.')
parser.add_argument('--lr', type=float, default=3e-5, help='The learning rate.')
parser.add_argument('--wd', type=float, default=0, help='The learning rate.')

parser.add_argument('--pre_epochs', type=int, default=3, help='The learning rate.')
parser.add_argument('--sim_score', type=str, default='cos', help='The Metrics for calculating similarity score.')

parser.add_argument('--num_workers', type=int, default=10, help='The number of workers.')

parser.add_argument('--ce_source', type=float, default=1.0, help='Weight of the cross entropy loss.')
parser.add_argument('--ce_target', type=float, default=1.0, help='Weight of the cross entropy loss.')
parser.add_argument('--scale', type=float, default=5, help='Scale up the cosine distance.')

parser.add_argument('--log_interval', type=int, default=50, help='The log-print interval.')
parser.add_argument('--test_interval', type=int, default=1, help='The evaluation interval.')

parser.add_argument("--pretrained_model", type=str, default="checkpoints/vit_base_patch16_224.pth",
                    help='pretrained ViT model')

parser.add_argument('--resume', type=str, default="", help='Resume the training')
parser.add_argument('--device', type=str, default='0', help='gpu index')
parser.add_argument('--keep_token', type=int, default=40, help='number of shots per class')
parser.add_argument('--pool', type=str, default='mean', help='average pooling or max pooling for feature maps')
parser.add_argument('--kl_t', type=float, default=20, help='kd temperature')
parser.add_argument('--kl', type=float, default=1, help='kld efficient')
parser.add_argument('--att_dec', type=float, default=0.3, help='att decorrelation efficient')
parser.add_argument('--patch_cls', type=float, default=3, help='token cls loss weight')
parser.add_argument('--replace_n', type=float, default=1, help='token cls loss weight')
parser.add_argument('--mse', type=float, default=0.0, help='Weight of the mse loss.')

parser.add_argument('--schedule_step_size', type=int, default=5, help='Learning rate schedule step size.')
parser.add_argument('--schedule_gamma', type=float, default=0.8, help='Learning rate schedule step gamma.')

parser.add_argument('--beta', type=float, default=0.5, help='optimizer beta1.')


args = parser.parse_args()
args.data_root = args.data_root + "x-{}-data-image.pth".format(args.dataset)
args.log_dir = "./logs/ViT-" + args.dataset + "-" + args.exp_name
if args.manual_seed is None or args.manual_seed < 0:
    args.manual_seed = random.randint(1, 100000)
assert args.log_dir is not None, 'The log_dir argument can not be None.'
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.autograd.set_detect_anomaly(True)


def train_model(args, loader, semantics, unseen_semantics, transformer,  optimizer, logger, epoch_str, epoch):
    batch_time, Xlosses, CElosses, BCElosses, ATTlosses, KLlosses, MSELosses, accs, token_accs, end = AverageMeter(), \
                                                        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), \
                                                AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()
    transformer.train()

    loader.dataset.set_return_label_mode('new')
    loader.dataset.set_return_img_mode('original')

    semantics = semantics.cuda()
    mse = torch.nn.MSELoss()
    bce= torch.nn.BCELoss()
    kld = torch.nn.KLDivLoss()

    for batch_idx, (img_feat, targets, idx) in enumerate(loader):

        batch = targets.shape[0]  # assume train and val has the same amount
        source_att_fm, pruned_att_fm, patch_labels, patch_pred = transformer(img_feat.cuda(), epoch=epoch)

        mse_loss = args.mse * mse(source_att_fm, semantics[targets]) + \
                   args.mse * mse(pruned_att_fm, semantics[targets])

        bce_loss = bce(patch_pred, patch_labels)
        cos_source = torch.einsum('bd,nd->bn', source_att_fm, semantics)
        cos_target = torch.einsum('bd,nd->bn', pruned_att_fm, semantics)
        kl_loss = kld(F.log_softmax(cos_source / args.kl_t, dim=1),
                        F.softmax(cos_target / args.kl_t, dim=1)) * args.kl_t * args.kl_t
        ce_loss = args.ce_source * F.cross_entropy(cos_source * args.scale, targets.cuda()) + \
                  args.ce_target * F.cross_entropy(cos_target * args.scale, targets.cuda())

        query = pruned_att_fm.T
        loss_att = 0
        for key in args.att_group:
            group = args.att_group[key]
            proto_each_group = query[group]  # g1 * v
            channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
            loss_att += channel_l2_norm.mean()
        loss_att = loss_att.float()/len(args.att_group) * args.att_dec
        loss = args.patch_cls * bce_loss + \
               args.kl * kl_loss + \
               ce_loss + \
               args.att_dec * loss_att + \
               mse_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)
        optimizer.step()

        Xlosses.update(loss.item(), batch)
        CElosses.update(ce_loss.item(), batch)
        BCElosses.update(bce_loss.item(), batch)
        ATTlosses.update(loss_att.item(), batch)
        KLlosses.update(kl_loss.item(), batch)
        MSELosses.update(mse_loss.item(), batch)

        predict_labels = torch.argmax(cos_target, dim=1)
        predict_tokens = patch_pred > 0.5
        with torch.no_grad():
            accuracy = (predict_labels.cpu() == targets).float().mean().item()
            accs.update(accuracy * 100, batch)
            accuracy = (predict_tokens == patch_labels).float().mean().item()
            token_accs.update(accuracy * 100, batch)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0 or batch_idx + 1 == len(loader):
            Tstring = 'TIME[{batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(batch_time=batch_time)
            Sstring = '{:} [{:}] [{:03d}/{:03d}]'.format(time_string(), epoch_str, batch_idx, len(loader))
            Astring = 'loss={:.7f} ({:.5f}), ce={:.7f} ({:.5f}), kl={:.7f} ({:.5f}), att={:.7f} ({:.5f}), ' \
                      'bce={:.7f} ({:.5f}), mse={:.7f} ({:.5f})' \
                      'acc@1={:.1f} ({:.1f}) acc_t={:.1f} ({:.1f})'.format( Xlosses.val, Xlosses.avg,
                                                      CElosses.val, CElosses.avg,
                                                      KLlosses.val, KLlosses.avg,
                                                      ATTlosses.val, ATTlosses.avg,
                                                      BCElosses.val, BCElosses.avg,
                                                      MSELosses.val, MSELosses.avg,
                                                      accs.val, accs.avg, token_accs.val, token_accs.avg)

            logger.print('{:} {:} {:} B={:},'.format(Sstring, Tstring, Astring, batch, ))


def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = Logger(args.log_dir, args.manual_seed)
    logger.print('args :\n{:}'.format(args))
    logger.print('PyTorch: {:}'.format(torch.__version__))


    torch.backends.cudnn.deterministic = True
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)

    logger.print('Start Main with this file : {:}'.format(__file__))
    graph_info = torch.load(Path(args.data_root))

    # prepare data loaders
    batch_size = args.bs

    args.att_group = get_attr_group(args.dataset)
    train_dataset = ZSLDataset(args, graph_info, 'train', feature=False, dataset=args.dataset)

    test_unseen_dataset = ZSLDataset(args, graph_info, 'test-unseen', feature=False, dataset=args.dataset)
    test_seen_dataset = ZSLDataset(args, graph_info, 'test-seen', feature=False, dataset=args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, drop_last=True) #xargs.num_workers
    test_seen_loader = DataLoader(test_seen_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_seen_dataset.set_return_img_mode('original')
    test_unseen_dataset.set_return_img_mode('original')

    logger.print('train-dataset       : {:}'.format(train_dataset))
    logger.print('test-seen-dataset   : {:}'.format(test_seen_dataset))
    logger.print('test-unseen-dataset : {:}'.format(test_unseen_dataset))

    features = graph_info['ori_attributes'].float().cuda()

    temp_norm = torch.norm(features, p=2, dim=1).unsqueeze(1).expand_as(features)
    features = features.div(temp_norm + 1e-5)

    train_features = features[graph_info['train_classes'], :]
    test_features = features[graph_info['unseen_classes'], :]
    logger.print('feature-shape={:}, train-feature-shape={:}'.format(list(features.shape), list(train_features.shape)))


    transformer = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None,
                              num_classes=args.attribute_dim, replace_n=args.replace_n, keep_token=args.keep_token,
                                    pool=args.pool, sim_score=args.sim_score , dataset=args.dataset)

    state = torch.load("/home/zhi/Projects/SCViP_ZSL/vit_base_patch16_224.pth")


    classifier_name = 'head'
    del state[classifier_name + '.weight']
    del state[classifier_name + '.bias']
    transformer.load_state_dict(state, strict=False)

    transformer.cuda()

    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_accs = checkpoint['best_accs']
        best_stacked_accs = checkpoint['best_stacked_accs']
        transformer.load_state_dict(checkpoint['network'])
        logger.print('load checkpoint from {:}'.format(args.resume))
        transformer.eval()
        with torch.no_grad():
            logger.print('-----start evaluation--------')
            xinfo = {'train_classes': graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}
            evaluate_all_dual(checkpoint['epoch'], test_unseen_loader, test_seen_loader, features, transformer,
                                             xinfo, best_accs, best_stacked_accs, logger, args.scale)
        return
    else:
        start_epoch, best_accs, best_stacked_accs = 0, {'train': -1, 'xtrain': -1, 'zs': -1, 'gzs-seen': -1,
                            'gzs-unseen': -1, 'gzs-H': -1, 'best-info': None}, {'train': -1, 'xtrain': -1, 'zs': -1,
                            'gzs-seen': -1, 'gzs-unseen': -1, 'gzs-H': -1, 'best-info': None}

    epoch_time, start_time = AverageMeter(), time.time()

    params_to_update = []
    params_names = []
    for name, param in transformer.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            params_names.append(name)


    optimizer = torch.optim.Adam(list(transformer.head_1.parameters()) +
                                 list(transformer.token_cls.parameters()) +
                                 list(transformer.v2p.parameters()) +
                                 list(transformer.p2t.parameters()), lr=args.pre_lr, betas=(args.beta, 0.999), weight_decay=args.wd)


    for iepoch in range(start_epoch, args.epochs):

        time_str = convert_secs2time(epoch_time.val * (args.epochs - iepoch), True)
        epoch_str = '{:03d}/{:03d}'.format(iepoch, args.epochs)

        if iepoch > args.pre_epochs:
            current_lr = args.lr * (args.schedule_gamma ** (iepoch // args.schedule_step_size)) #1e-4
            optimizer = torch.optim.Adam(list(transformer.parameters()), lr=current_lr, betas=(args.beta, 0.999), weight_decay=args.wd)
            logger.print('Train the {:}-th epoch, {:}, LR={:1.6f} ~ {:1.6f}'.format(epoch_str, time_str, (current_lr), (current_lr)))
        else:
            logger.print('Train the {:}-th epoch, {:}, LR={:1.6f}'.format(epoch_str, time_str, args.pre_lr))


        train_model(args, train_loader, train_features, test_features, transformer, optimizer, logger, epoch_str, iepoch)

        if iepoch % args.test_interval == 0 or iepoch == args.epochs - 1:
            transformer.eval()
            with torch.no_grad():
                xinfo = {'train_classes': graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}

                logger.print('-----test--------')
                better_model = evaluate_all_dual(epoch_str, test_unseen_loader, test_seen_loader, features, transformer, xinfo,
                                  best_accs, best_stacked_accs, logger, args.scale)
            transformer.train()

        # save the info
        if better_model:
            info = {'epoch': iepoch,
                    'args': deepcopy(args),
                    'finish': iepoch + 1 == args.epochs,
                    'best_accs': best_accs,
                    'best_stacked_accs': best_stacked_accs,
                    'semantic_lists': None,
                    'network': transformer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
            try:
                args_path = args.log_dir + "/ckp-step:{}-{}-mse:{}-ce1:{}{}-kl:{}-bs:{}-".format(
                    args.schedule_step_size, args.schedule_gamma, args.mse, args.ce_source, args.ce_target, args.kl, args.bs)
                files2remove = glob.glob(args_path + '*')
                for _i in files2remove:
                    os.remove(_i)
                ckp_path = args_path + "{:.1f}-{:.1f}-{:.1f}.pth".format(best_stacked_accs['gzs-unseen'],
                    best_stacked_accs['gzs-seen'], best_stacked_accs['gzs-H'])
                torch.save(info, ckp_path)
                logger.print('--->>> :: ckp saved into {:}.\n'.format(ckp_path))
            except PermissionError:
                print('unsuccessful write log')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.close()


if __name__ == '__main__':
    main(args)
