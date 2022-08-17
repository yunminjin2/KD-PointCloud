import argparse
from calendar import EPOCH
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models_bid_pointconv import PointConvBidirection
from models_bid_lighttoken_res import PointConvBidirection as PointConvStudentModel
from loss_functions import  multiScaleLoss, attentiveImitationLoss, biDirectionLoss, loss_fn_kd_2, loss_fn_ht, biDirection_loss_ht
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/lighttoken_res4_bidirht' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('train.py', log_dir))
    os.system('cp %s %s' % ('config_train.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    teacher_model_path = args.ckpt_dir + args.teacher_model
    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Augmentation(args.aug_together,
                                            args.aug_pc2,
                                            args.data_process,
                                            args.num_points),
        num_points=args.num_points,
        data_root = args.data_root,
        full=args.full       
    )
    print("Load Train Dataset")
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    print("Load Train Loader")
    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root = args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    print("Load Val Loader")
    print(args)
    blue = lambda x: '\033[94m' + x + '\033[0m'
    t_model = PointConvBidirection()
    t_model.load_state_dict(torch.load(teacher_model_path))
    st_model = PointConvStudentModel()

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True 
        t_model.cuda(device_ids[0])
        t_model = torch.nn.DataParallel(t_model, device_ids = device_ids)
        st_model.cuda(device_ids[0])
        st_model = torch.nn.DataParallel(st_model, device_ids = device_ids)
    else:
        t_model.cuda()
        st_model.cuda()


    if args.pretrain is not None:
        st_model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    pretrain = args.pretrain 
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0 
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(st_model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(st_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
                
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch = init_epoch - 1)
    LEARNING_RATE_CLIP = 5e-5 
    
    # Fast Configuration
    FROM = 20
    UNTIL = 400
    FAST_RATIO = 5
    # Get max, min teacher loss
    # _, _, t_history = eval_sceneflow(t_model, train_loader)
    history = defaultdict(lambda: list())
    best_epe = 1000.0
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # gamma = distillSchedule(epoch, _upto=30)
        # print("gamma: ", gamma)
        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            pos1, pos2, norm1, norm2, flow, _ = data  
            #move to cuda 
            pos1 = pos1.cuda()
            pos2 = pos2.cuda() 
            norm1 = norm1.cuda()
            norm2 = norm2.cuda()
            flow = flow.cuda() 
            
            t_model.eval()
            with torch.no_grad():
                t_pred_flows, t_fps_pc1_idxs, t_fps_pc2_idxs, t_pc1, t_pc2, t_feat1s, t_feat2s = t_model(pos1, pos2, norm1, norm2)
            st_model.train()         
            pred_flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, feat1s, feat2s = st_model(pos1, pos2, norm1, norm2)
            
            # loss = loss_fn_kd_2(pred_flows, fps_pc1_idxs, flow, t_pred_flows, t_fps_pc1_idxs, 0.3)
            # loss = attentiveImitationLoss(pred_flows, fps_pc1_idxs, flow, t_pred_flows, t_fps_pc1_idxs, t_history, 0.3)
            # loss = biDirectionLoss(pred_flows, fps_pc1_idxs, fps_pc2_idxs, flow, t_pred_flows,  t_fps_pc1_idxs, 0.3, 1.0, 0.9)
            loss = biDirection_loss_ht(pred_flows, feat1s, feat2s, fps_pc1_idxs, fps_pc2_idxs, flow, t_pred_flows, t_feat1s, t_feat2s, t_fps_pc1_idxs, t_fps_pc2_idxs, 0.3, 0.8,  layer=3)
            # loss = loss_fn_ht(pred_flows, feat1s, fps_pc1_idxs, fps_pc2_idxs, flow, t_pred_flows, t_feat1s, t_fps_pc1_idxs, 0.3, layer=3)

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

            total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), train_loss)
        print(str_out)
        logger.info(str_out)
        if epoch%FAST_RATIO != 0 and FROM < epoch and epoch < UNTIL:
            continue
        else:
            eval_epe3d, eval_loss,_ = eval_sceneflow(st_model.eval(), val_loader)
            str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, eval_loss)
            print(str_out)
            logger.info(str_out)

            if eval_epe3d < best_epe:
                best_epe = eval_epe3d
                if args.multi_gpu is not None:
                    torch.save(st_model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
                else:
                    torch.save(st_model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
                logger.info('Save model ...')
                print('Save model ...')
            print('Best epe loss is: %.5f'%(best_epe))
            logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader):
    metrics = defaultdict(lambda:list())
    history = []
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, _ = data  
        
        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        with torch.no_grad():
            pred_flows, fps_pc1_idxs, _, _, _, _, _ = model(pos1, pos2, norm1, norm2)

            eval_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
            history.append(eval_loss)
            epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim = 2).mean()

        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())

    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_eval = np.mean(metrics['eval_loss'])

    return mean_epe3d, mean_eval, history

def analyzing():
    from pointconv_util import index_points_gather as index_points

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Augmentation(args.aug_together,
                                            args.aug_pc2,
                                            args.data_process,
                                            args.num_points),
        num_points=args.num_points,
        data_root = args.data_root,
        # full=args.full        for flying
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root = args.data_root
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    

    teacher_model_path = args.ckpt_dir + args.teacher_model
    t_model = PointConvBidirection()
    t_model.load_state_dict(torch.load(teacher_model_path))
    t_model.cuda()

    st_model = PointConvBidStudentModel()
    st_model.cuda()


    for i, data in enumerate(train_loader, 0):
        pos1, pos2, norm1, norm2, flow, _ = data  
        
        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()

        t_model.eval()
        with torch.no_grad():
            t_pred_flows, t_fps_pc1_idxs, t_fps_pc2_idxs, t_pc1, t_pc2, _, _ = t_model(pos1, pos2, norm1, norm2)
        st_model.train()
        pred_flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, _, _ = st_model(pos1, pos2, norm1, norm2)
        gt_flows = [flow]
        for i in range(1, len(t_fps_pc1_idxs) + 1):
            fps_idx = t_fps_pc1_idxs[i - 1]
            sub_gt_flow = index_points(gt_flows[-1], fps_idx)
            gt_flows.append(sub_gt_flow)

        pos1 = pos1.permute(0, 2, 1)
        pos2 = pos2.permute(0, 2, 1)
        flow = flow.permute(0, 2, 1)
 
        for i in range(0, 50):
                
            print(pos1[0][0][i], pos1[0][1][i], pos1[0][2][i], " -- ", t_pc1[1][0][0][i], t_pc1[1][0][1][i], t_pc1[1][0][2][i], "[", t_fps_pc1_idxs[0][0][i], "]")
            print(pos2[0][0][i], pos2[0][1][i], pos2[0][2][i], " -- ", t_pc2[1][0][0][i], t_pc2[1][0][1][i], t_pc2[1][0][2][i], "[", t_fps_pc2_idxs[0][0][i], "]")

            print(flow[0][0][i], flow[0][1][i], flow[0][2][i], " -- ", t_pred_flows[0][0][0][i], t_pred_flows[0][0][1][i], t_pred_flows[0][0][2][i])

        break
# ______________________    ____

def distillSchedule(cur, _from=0, _upto=100):
    if cur > _upto:
        return 0
    elif cur < _from:
        return 1
    return 1 - (cur + 1)/(_upto + 1)

def to_np(t):
    result = []
    for i in t:
        result.append(i.cpu().detach().numpy())

    return result       

if __name__ == '__main__':
    main()
    # analyzing()
    
