import os
from datetime import datetime

import imageio
import numpy as np
import scipy.io as sio
import sklearn.metrics as metrics
import torch
#import tensorboardX
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from data import CreateSrcDataLoader, CreateTrgDataLoader
from model import CreateModel
from options.train_options import TrainOptions
from utils import FDA_source_to_target
from utils.timer import Timer

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
CS_weights = np.array( (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32 )
CS_weights = torch.from_numpy(CS_weights)


def main():
    opt = TrainOptions()
    args = opt.initialize()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {
        'iter time' : Timer(),
        'data loading': Timer(),
        'fda': Timer(),
        'optimizer step': Timer()
    }

    model_name = args.source + '_to_' + args.target

    experiment_name = f"{args.experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    snapshot_dir = os.path.join(args.snapshot_dir, experiment_name)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
        os.makedirs(os.path.join(snapshot_dir, 'logs'))
    opt.print_options(args)

    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)

    model, optimizer = CreateModel(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    model.cuda()

    # losses to log
    loss = ['loss_seg_src', 'loss_seg_trg']
    loss_train = 0.0
    loss_val = 0.0
    loss_train_list = []
    loss_val_list = []

    mean_img = torch.zeros(1, 1)
    class_weights = Variable(CS_weights).cuda()

    # Create TensorBoard writer
    tensorboard_writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()                                                        # zero grad

        _t['data loading'].tic()

        src_img, src_lbl, _, _ = sourceloader_iter.next()                            # new batch source
        trg_img, trg_lbl, _, _ = targetloader_iter.next()                            # new batch target

        scr_img_copy = src_img.clone()

        _t['data loading'].toc()

        if mean_img.shape[-1] < 2:
            B, C, H, W = src_img.shape
            mean_img = IMG_MEAN.repeat(B,1,H,W)

            # print("Mean image adjusted")      

        #-------------------------------------------------------------------#

        _t['fda'].tic()
        # 1. source to target, target to target
        src_in_trg = FDA_source_to_target( src_img, trg_img, L=args.LB )            # src_lbl
        trg_in_trg = trg_img
        _t['fda'].toc()

        # print(f"Mean image shape: {mean_img.shape}")

        # print(f"Source image shape: {src_img.shape}")
        # print(f"Source in target image shape: {src_in_trg.shape}")


        # print(f"Target image shape: {trg_img.shape}")
        # print(f"Target in target image shape: {trg_in_trg.shape}")
        

        # 2. subtract mean
        src_img = src_in_trg.clone() - mean_img                                 # src, src_lbl
        trg_img = trg_in_trg.clone() - mean_img                                 # trg, trg_lbl

        #-------------------------------------------------------------------#

        _t['optimizer step'].tic()

        # evaluate and update params #####
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda() # to gpu
        src_seg_score = model(src_img, lbl=src_lbl, weight=class_weights, ita=args.ita)      # forward pass
        loss_seg_src = model.loss_seg                                                # get loss
        loss_ent_src = model.loss_ent

        # get target loss, only entropy for backpro
        trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda() # to gpu
        trg_seg_score = model(trg_img, lbl=trg_lbl, weight=class_weights, ita=args.ita)      # forward pass
        loss_seg_trg = model.loss_seg                                                # get loss
        loss_ent_trg = model.loss_ent

        triger_ent = 1.0 if i > args.switch2entropy else 0.0

        loss_all = loss_seg_src + triger_ent * args.entW * loss_ent_trg     # loss of seg on src, and ent on s and t

        loss_all.backward()
        optimizer.step() 

        _t['optimizer step'].toc()

        loss_train += loss_seg_src.detach().cpu().numpy()
        loss_val   += loss_seg_trg.detach().cpu().numpy()

        if (i+1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save( model.state_dict(), os.path.join(snapshot_dir, '%s_' % (args.source) + str(i+1) + '.pth') )
            
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print('[it %d][src seg loss %.4f][trg seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, loss_seg_trg.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff) )

            # print(f"Image shape: {src_img.shape}")
            imageio.imwrite(f"{args.tempdata}/src_img_{i}.png", src_img.cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1], format="png")
            imageio.imwrite(f"{args.tempdata}/trg_img_{i}.png", trg_img.cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1], format="png")
            # print(trg_seg_score.shape)

            # Source labels (ground truth), as flattened class array
            src_lbl_np = src_lbl.cpu().numpy().flatten()
            src_seg_score_np = src_seg_score.argmax(dim=1).cpu().numpy().flatten()

            accuracy_src = metrics.jaccard_score(y_true=src_lbl_np, y_pred=src_seg_score_np, average="macro")

            # Target labels (ground truth), as flattened class array
            trg_lbl_np = trg_lbl.cpu().numpy().flatten()
            trg_seg_score_np = trg_seg_score.argmax(dim=1).cpu().numpy().flatten()

            accuracy_val = metrics.jaccard_score(y_true=trg_lbl_np, y_pred=trg_seg_score_np, average="macro")

            # imageio.imwrite(f"{args.tempdata}/result_{i}.png", trg_seg_score.cpu().detach().numpy()[0].transpose((1, 2, 0))[:, :, ::-1], format="png")
            # sio.savemat(args.tempdata, {'src_img':src_img.cpu().numpy(), 'trg_img':trg_img.cpu().numpy()})

            loss_train /= args.print_freq
            loss_val   /= args.print_freq

            # Write losses to TensorBoard
            tensorboard_writer.add_scalar("Loss/Train", loss_train, i)
            tensorboard_writer.add_scalar("Loss/Val", loss_val, i)
            tensorboard_writer.add_scalar("Accuracy/Train", accuracy_src, i)
            tensorboard_writer.add_scalar("Accuracy/Val", accuracy_val, i)

            # Write source and target images to Tensorboard
            # tensorboard_writer.add_images("Images/Train", torch.cat((src_img.cpu()[None, 0, ...], trg_img.cpu()[None, 0, ...])), i)

            # loss_train_list.append(loss_train)
            # loss_val_list.append(loss_val)
            # sio.savemat( args.matname, {'loss_train':loss_train_list, 'loss_val':loss_val_list} )
            loss_train = 0.0
            loss_val = 0.0

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

    print("Average time per iteration part:")
    for k, v in _t.items():
        print(f'{k}: {v.average_time:.3f} s')

if __name__ == '__main__':
    main()

