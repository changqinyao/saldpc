import cv2
import os

os.environ["DHF1K_DATA_DIR"] = '/home/ubuntu/data/annonation/'
os.environ["SALICON_DATA_DIR"] = '/home/ubuntu/data/salicon/'

import datetime
import numpy as np

from args import get_training_parser
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from data_loader import DHF1K_frames, Hollywood_frames,SALICONDataset

from model import salSEPC,salSEPCema
import metric
from utils import nss,kld_loss,corr_coeff

"""
Be sure to check the name of args.new_model before running
"""
frame_size = (192, 256)  # original shape is (360, 640, 3)
# learning_rate = 0.0000001 # Added another 0 for hollywood
decay_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
# epochs = 7+1 #+3+2
plot_every = 1
clip_length = 8

# temporal = True
# RESIDUAL = False
# args.dropout = True
SALGAN_WEIGHTS = 'model_weights/salgan_salicon.pt'  # JuanJo's weights
# CONV_LSTM_WEIGHTS = './SalConvLSTM.pt' #These are not relevant in this problem after all, SalGAN was trained on a range of 0-255, the ConvLSTM was trained on a 0-1 range so they are incompatible.
LEARN_ALPHA_ONLY = False
# EMA_LOC_2 = 54
# PROCESS = 'parallel'
# Parameters
params_train = {'batch_size': 1,
           'shuffle':True,
          # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

params_val = {'batch_size': 1,
          # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

params_img={'batch_size': 4,
             'shuffle':True,
          # number of images / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

params_img_val={'batch_size': 1,
          # number of images / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self):
        self.nss = []
        self.kl = []
        self.cc = []
        self.auc = []
        self.loss=[]

    def update(self,outputs,sal,fix,loss):
        nss = metric.NSS(outputs, fix)
        cc=metric.CC(outputs,sal)
        self.nss.append(nss)
        self.loss.append(loss)
        self.cc.append(cc)


    def get_metrics(self):
        nss=np.nanmean(self.nss)
        cc=np.nanmean(self.cc)
        loss=np.nanmean(self.loss)
        return cc,nss,loss



def loss_sequences(pred_seq, sal_seq, fix_seq, metrics=['kld','nss','cc']):
    """
    Compute the training losses
    """

    losses = []
    for this_metric in metrics:
        if this_metric == 'kld':
            losses.append(kld_loss(pred_seq, sal_seq))
        if this_metric == 'nss':
            losses.append(nss(pred_seq, fix_seq))
        if this_metric == 'cc':
            losses.append(corr_coeff(pred_seq, sal_seq))
    return losses


def main(args):
    # =================================================
    # ================ Data Loading ===================

    # Expect Error if either validation size or train size is 1
    if args.dataset == "DHF1K":
        print("Commencing training on dataset {}".format(args.dataset))
        train_set = DHF1K_frames(
            root_path=args.src,
            load_gt=True,
            number_of_videos=int(args.end),
            starting_video=int(args.start),
            clip_length=clip_length,
            resolution=frame_size,
            val_perc=args.val_perc,
            split="train")
        print("Size of train set is {}".format(len(train_set)))
        train_loader = data.DataLoader(train_set, **params_train)

        if args.val_perc > 0:
            val_set = DHF1K_frames(
                root_path=args.src,
                load_gt=True,
                number_of_videos=int(args.end),
                starting_video=int(args.start),
                clip_length=clip_length,
                resolution=frame_size,
                val_perc=args.val_perc,
                split="validation")
            print("Size of validation set is {}".format(len(val_set)))
            val_loader = data.DataLoader(val_set, **params_val)

    elif args.dataset == "Hollywood-2" or args.dataset == "UCF-sports":
        print("Commencing training on dataset {}".format(args.dataset))
        train_set = Hollywood_frames(
            root_path="/imatge/lpanagiotis/work/{}/training".format(args.dataset),
            # root_path = "/home/linardosHollywood-2/training",
            clip_length=clip_length,
            resolution=frame_size,
            load_gt=True)
        video_name_list = train_set.video_names()  # match an index to the sample video name
        train_loader = data.DataLoader(train_set, **params)
    elif args.dataset == "salicon":
        train_set = SALICONDataset()
        print("Size of train set is {}".format(len(train_set)))
        train_loader = data.DataLoader(train_set, **params_img)
        val_set = SALICONDataset(phase='val')
        print("Size of validation set is {}".format(len(val_set)))
        val_loader = data.DataLoader(val_set, **params_img_val)

    else:
        print('Your model was not recognized. Check the name again.')
        exit()
    # =================================================
    # ================ Define Model ===================

    # The seed pertains to initializing the weights with a normal distribution
    # Using brute force for 100 seeds I found the number 65 to provide a good starting point (one that looks close to a saliency map predicted by the original SalGAN)
    temporal = True
    if 'CLSTM56' in args.new_model:
        model = SalGANmore.SalGANplus(seed_init=65, freeze=args.thaw)
        print("Initialized {}".format(args.new_model))
    elif 'CLSTM30' in args.new_model:
        model = SalGANmore.SalCLSTM30(seed_init=65, residual=args.residual, freeze=args.thaw)
        print("Initialized {}".format(args.new_model))
    elif 'SalBCE' in args.new_model:
        model = SalGANmore.SalGAN()
        print("Initialized {}".format(args.new_model))
        temporal = False
    elif 'EMA' in args.new_model:
        if args.double_ema != False:
            model = salSEPCema(alpha=0.3)
            print("Initialized {}".format(args.new_model))
        else:
            if args.dataset=='DHF1K':
                model = salSEPCema(alpha=None,backbone_name='res2net')
            if args.dataset=='salicon':
                model = salSEPC(backbone_name='res2net')
            print("Initialized {} with residual set to {} and dropout set to {}".format(args.new_model, args.residual,
                                                                                        args.dropout))
    else:
        print("Your model was not recognized, check the name of the model and try again.")
        exit()
    # criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    # bceloss = nn.BCELoss()
    # nssloss =
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.99, eps=1e-08, momentum=momentum, weight_decay=weight_decay)
    # start
    bceloss = nn.BCELoss()

    if args.thaw:
        # Load only the unfrozen part to the optimizer

        if args.new_model == 'SalGANplus.pt':
            optimizer = torch.optim.Adam(
                [{'params': model.Gates.parameters()}, {'params': model.final_convs.parameters()}], args.lr,
                betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

        elif 'SalCLSTM30' in args.new_model:
            optimizer = torch.optim.Adam([{'params': model.Gates.parameters()}], args.lr, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=weight_decay)

    else:
        # optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        pg0,pg1,pg2=[],[],[]
        if args.alpha is None:
            for k, v in model.named_parameters():
                # v.requires_grad = True
                if k=='alpha':
                    continue
                else:
                    if '.bias' in k:
                        pg2.append(v)  # biases
                    elif '.weight' in k and '.bn' not in k:
                        pg1.append(v)  # apply weight decay
                    else:
                        pg0.append(v)  # all else
            optimizer = torch.optim.Adam(pg0, lr=args.lr, betas=(0.937, 0.999))
            optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
            optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
            if hasattr(model, 'alpha'):
                optimizer.add_param_group({'params': model.alpha, 'lr': 0.1})
            del pg0, pg1, pg2
            # optimizer = torch.optim.Adam([
            #     {'params': model.parameters(), 'lr': args.lr, 'weight_decay': weight_decay},
            #     {'params': model.alpha, 'lr': 0.1}])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.salgan.parameters(), 'lr': args.lr, 'weight_decay': weight_decay}])

        if LEARN_ALPHA_ONLY:
            optimizer = torch.optim.Adam([{'params': [model.alpha]}], 0.1)


    scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=3)
    if args.pt_model == None:
        # In truth it's not None, we default to SalGAN or SalBCE (JuanJo's)weights
        # By setting strict to False we allow the model to load only the matching layers' weights
        if SALGAN_WEIGHTS == 'model_weights/gen_model.pt':
            model.salgan.load_state_dict(torch.load(SALGAN_WEIGHTS), strict=False)
        else:
            model.salgan.load_state_dict(torch.load(SALGAN_WEIGHTS)['state_dict'], strict=False)

        start_epoch = 1

    else:
        # Load an entire pretrained model
        # checkpoint = load_weights(model, args.pt_model)
        # model.load_state_dict(checkpoint, strict=False)
        # start_epoch = torch.load(args.pt_model, map_location='cpu')['epoch']
        # #optimizer.load_state_dict(torch.load(args.pt_model, map_location='cpu')['optimizer'])
        #
        # print("Model loaded, commencing training from epoch {}".format(start_epoch))
        start_epoch = 0
        pass

    dtype = torch.FloatTensor
    if args.use_gpu == 'parallel' or args.use_gpu == 'gpu':
        assert torch.cuda.is_available(), \
            "CUDA is not available in your machine"

        if args.use_gpu == 'parallel':
            model = nn.DataParallel(model).cuda()
        elif args.use_gpu == 'gpu':
            model = model.cuda()
        dtype = torch.cuda.FloatTensor
        cudnn.benchmark = True  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        # criterion = criterion.cuda()
        bceloss=bceloss.cuda()
    # =================================================
    # ================== Training =====================

    # 加载静态
    checkpoint = torch.load('salsepcbce.pt')
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    train_losses = []
    val_nsses = []
    starting_time = datetime.datetime.now().replace(microsecond=0)
    print("Training started at : {}".format(starting_time))
    max_val_nss=float('-inf')

    n_iter = 0
    # if "EMA" in args.new_model:
    #    print("Alpha value started at: {}".format(model.alpha))

    for epoch in range(start_epoch, args.epochs + 1):

        # try:
            # adjust_learning_rate(optimizer, epoch, decay_rate) #Didn't use this after all
            # train for one epoch
        if args.dataset == "salicon":
            train_loss, n_iter, optimizer = train_img(train_loader, model, bceloss, optimizer, epoch, n_iter,
                                                  args.use_gpu, args.double_ema, args.thaw, temporal, dtype)
            #
            print("Epoch {}/{} done with train loss {}\n".format(epoch, args.epochs, train_loss))

            if args.val_perc > 0:
                print("Running validation..")
                with torch.no_grad():
                    nss,loss,n_iter, optimizer = val_img(val_loader, model, bceloss, optimizer, epoch, n_iter,
                                                      args.use_gpu, args.double_ema, args.thaw, temporal, dtype)
                print("Epoch{}\tValidation loss: {}\t nss : {}".format(epoch,loss,nss))
                scheduler.step(nss)

            if epoch % plot_every == 0:
                train_losses.append(train_loss.cpu())
                if args.val_perc > 0:
                    val_nsses.append(nss)
            if nss>max_val_nss:
                max_val_nss=nss
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.cpu().state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_nss':max_val_nss
                }, 'salsepcbce' + ".pt")

            if args.use_gpu == 'parallel':
                model = nn.DataParallel(model).cuda()
            elif args.use_gpu == 'gpu':
                model = model.cuda()
            else:
                pass

            """
            else:

                print("Training on whole set")
                train_loss, n_iter, optimizer = train(whole_loader, model, criterion, optimizer, epoch, n_iter)
                print("Epoch {}/{} done with train loss {}".format(epoch, args.epochs, train_loss))
            """

        # except RuntimeError:
        #     print("A memory error was encountered. Further training aborted.")
        #     epoch = epoch - 1
        #     break
        if args.dataset == "DHF1K":
            train_loss, n_iter, optimizer = train(train_loader, model, bceloss, optimizer, epoch, n_iter,
                                                  args.use_gpu, args.double_ema, args.thaw, temporal, dtype)

            print("Epoch {}/{} done with train loss {}\n".format(epoch, args.epochs, train_loss))

            if args.val_perc > 0:
                print("Running validation..")
                cc,nss,val_loss = validate(val_loader, model,epoch, temporal, dtype)
                print("Validation loss: {}\t nss : {}\t cc:{}".format(val_loss,nss,cc))
                scheduler.step(nss)

            # if epoch % plot_every == 0:
            #     train_losses.append(train_loss.cpu())
            #     if args.val_perc > 0:
            #         val_nsses.append(nss)
            if nss>min_val_nss:
                min_val_nss=nss
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.cpu().state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_nss':min_val_nss
                }, args.new_model + ".pt")

            if args.use_gpu == 'parallel':
                model = nn.DataParallel(model).cuda()
            elif args.use_gpu == 'gpu':
                model = model.cuda()
            else:
                pass

            """
            else:

                print("Training on whole set")
                train_loss, n_iter, optimizer = train(whole_loader, model, criterion, optimizer, epoch, n_iter)
                print("Epoch {}/{} done with train loss {}".format(epoch, args.epochs, train_loss))
            """

        # except RuntimeError:
        #     print("A memory error was encountered. Further training aborted.")
        #     epoch = epoch - 1
        #     break

    print("Training of {} started at {} and finished at : {} \n Now saving..".format(args.new_model, starting_time,
                                                                                     datetime.datetime.now().replace(
                                                                                         microsecond=0)))
    # if "EMA" in args.new_model:
    #    print("Alpha value tuned to: {}".format(model.alpha))
    # ===================== #
    # ======  Saving ====== #

    # If I try saving in regular intervals I have to move the model to CPU and back to GPU.
    # torch.save({
    #     'epoch': epoch + 1,
    #     'state_dict': model.cpu().state_dict(),
    #     'optimizer': optimizer.state_dict()
    # }, args.new_model + ".pt")

    """
    hyperparameters = {
        'momentum' : momentum,
        'weight_decay' : weight_decay,
        'args.lr' : learning_rate,
        'decay_rate' : decay_rate,
        'args.epochs' : args.epochs,
        'batch_size' : batch_size
    }
    """

    if args.val_perc > 0:
        to_plot = {
            'epoch_ticks': list(range(start_epoch, args.epochs + 1, plot_every)),
            'train_losses': train_losses,
            'val_nsses': val_nsses
        }
        with open('to_plot.pkl', 'wb') as handle:
            pickle.dump(to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ===================

mean = lambda x: sum(x) / len(x)


def adjust_learning_rate(optimizer, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (decay_rate ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_weights(model, pt_model, device='cpu'):
    # Load stored model:
    temp = torch.load(pt_model, map_location=device)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.", "")
        checkpoint[new_key] = temp[key]

    return checkpoint


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(train_loader, model, criterion, optimizer, epoch, n_iter, use_gpu, double, thaw, temporal, dtype):
    # Switch to train mode
    model.train()

    if temporal and thaw:
        if use_gpu == 'parallel':
            # Unfreeze layers depending on epoch number
            optimizer = model.module.thaw(epoch,
                                          optimizer)  # When you wrap a model with DataParallel, the model.module can be seen as the model before it’s wrapped.

            # Confirm:
            model.module.print_layers()
        else:
            # Unfreeze layers depending on epoch number
            optimizer = model.thaw(epoch, optimizer)

            # Confirm:
            model.print_layers()

    video_losses = []
    print("Now commencing epoch {}".format(epoch))
    for i, video in enumerate(train_loader):
        """
        if i == 956 or i == 957:
            #some weird bug happens there
            continue
        """
        # print(type(video))
        accumulated_losses = []
        start = datetime.datetime.now().replace(microsecond=0)
        print("Number of clips for video {} : {}".format(i, len(video)))
        state = None  # Initially no hidden state
        for j, (clip, gtruths,gtruths_fix) in enumerate(video):

            n_iter += j

            # Reset Gradients
            optimizer.zero_grad()

            # Squeeze out the video dimension
            # [video_batch, clip_length, channels, height, width]
            # After transpose:
            # [clip_length, video_batch, channels, height, width]

            clip = Variable(clip.type(dtype).transpose(0, 1))
            gtruths = Variable(gtruths.type(dtype).transpose(0, 1))
            gtruths_fix=Variable(gtruths_fix.type(dtype).transpose(0, 1))

            if temporal and not double:
                # print(clip.size()) #works! torch.Size([5, 1, 1, 360, 640])
                loss3 = 0
                b_size=clip.size()[0]
                for idx in range(b_size):
                    # print(clip[idx].size())

                    # Compute output
                    state, saliency_map = model.forward(x=clip[idx],
                                                        prev_state=state)  # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones

                    saliency_map = saliency_map.squeeze(0)  # Target is 3 dimensional (grayscale image)
                    if saliency_map.size() != gtruths[idx].size():
                        # print(saliency_map.size())
                        # print(gtruths[idx].size())
                        a, b, c, _ = saliency_map.size()
                        saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()],
                                                 3)  # because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2

                    # Apply sigmoid before visualization
                    # logits will be whatever you have to rescale this

                    # Compute loss
                    loss = loss_sequences(saliency_map, gtruths[idx], gtruths_fix[idx])
                    loss3+=loss[0]-0.1*loss[1]-0.1*loss[2]


                # Keep score
                loss3=loss3/b_size
                accumulated_losses.append(loss3.data)

                # Compute gradient
                loss3.backward()

                # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
                nn.utils.clip_grad_norm_(model.parameters(), 10 * clip.size()[0])

                # Update parameters
                optimizer.step()

                # Repackage to avoid backpropagating further through time
                state = repackage_hidden(state)

            elif temporal and double:
                if state == None:
                    state = (None, None)
                loss = 0
                for idx in range(clip.size()[0]):
                    # print(clip[idx].size())

                    # Compute output
                    state, saliency_map = model.forward(input_=clip[idx], prev_state_1=state[0], prev_state_2=state[
                        1])  # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones

                    saliency_map = saliency_map.squeeze(0)  # Target is 3 dimensional (grayscale image)
                    if saliency_map.size() != gtruths[idx].size():
                        print(saliency_map.size())
                        print(gtruths[idx].size())
                        a, b, c, _ = saliency_map.size()
                        saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()],
                                                 3)  # because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2

                    # Apply sigmoid before visualization
                    # logits will be whatever you have to rescale this

                    # Compute loss
                    loss=loss_sequences(saliency_map, gtruths[idx],gtruths_fix[idx])
                    loss=loss[0]-0.1*loss[1]-0.1*loss[2]

                # Keep score
                accumulated_losses.append(loss.item())

                # Compute gradient
                loss.backward()

                # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
                nn.utils.clip_grad_norm_(model.parameters(), 10 * clip.size()[0])

                # Update parameters
                optimizer.step()

                # Repackage to avoid backpropagating further through time
                state = repackage_hidden(state)

            else:
                # print(type(clip))
                # print(clip.size())
                for idx in range(clip.size()[0]):
                    saliency_map = model.forward(clip[idx])
                    saliency_map = saliency_map.squeeze(0)
                    loss = criterion(saliency_map, gtruths[idx])
                    loss.backward()
                    optimizer.step()

                    accumulated_losses.append(loss.data)

            # Visualize some of the data
            if i % 100 == 0 and j == 5:

                # writer.add_image('Frame', clip[idx], n_iter)
                # writer.add_image('Gtruth', gtruths[idx], n_iter)

                post_process_saliency_map = (saliency_map - torch.min(saliency_map)) / (
                            torch.max(saliency_map) - torch.min(saliency_map))
                utils.save_image(post_process_saliency_map, "./log/smap{}_epoch{}.png".format(i, epoch))

                if epoch == 1:
                    print(saliency_map.max())
                    print(saliency_map.min())
                    print(gtruths[idx].max())
                    print(gtruths[idx].min())
                    print(post_process_saliency_map.max())
                    print(post_process_saliency_map.min())
                    utils.save_image(gtruths[idx], "./log/gt{}.png".format(i))
                # writer.add_image('Prediction', prediction, n_iter)

        end = datetime.datetime.now().replace(microsecond=0)
        print('Epoch: {}\tVideo: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, i, mean(accumulated_losses),
                                                                                     end - start))
        video_losses.append(mean(accumulated_losses))

    return (mean(video_losses), n_iter, optimizer)

def validate(val_loader, model,epoch, temporal, dtype):
    # switch to evaluate mode
    model.eval()
    summary=Meter()
    video_losses = []
    print("Now running validation..")
    for i, video in enumerate(val_loader):
        accumulated_losses = []
        state = None  # Initially no hidden state
        with torch.no_grad():
            for j, (clip, gtruths,gtruths_fix) in enumerate(video):

                clip = Variable(clip.type(dtype).transpose(0, 1), requires_grad=False)
                gtruths = Variable(gtruths.type(dtype).transpose(0, 1), requires_grad=False)
                gtruths_fix=Variable(gtruths_fix.type(dtype).transpose(0, 1), requires_grad=False)

                loss = 0
                for idx in range(clip.size()[0]):
                    # print(clip[idx].size()) needs unsqueeze
                    # Compute output
                    if temporal:
                        state, saliency_map = model.forward(clip[idx], state)
                    else:
                        saliency_map = model.forward(clip[idx])

                    saliency_map = saliency_map.squeeze(0)

                    # if saliency_map.size() != gtruths[idx].size():
                    #     a, b, c, _ = saliency_map.size()
                    #     saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()],
                    #                              3)  # because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2

                    # Compute loss
                    # nss = metric.CC(saliency_map.cpu().detach().numpy(), gtruths[idx].cpu().detach().numpy())
                    loss=loss_sequences(saliency_map, gtruths[idx],gtruths_fix[idx])

                    saliency_map,sal,fix=saliency_map.cpu().detach().numpy(), gtruths[idx].cpu().detach().numpy(), gtruths_fix[idx].cpu().detach().numpy()
                    summary.update(saliency_map,sal,fix, (loss[0]-0.1*loss[1]-0.1*loss[2]).item())

                if temporal:
                    state = repackage_hidden(state)

                # Keep score
                # accumulated_losses.append(loss.data)

            # video_losses.append(mean(accumulated_losses))
    cc,nss,loss=summary.get_metrics()

    return cc,nss,loss



def train_img(train_loader, model, criterion, optimizer, epoch, n_iter, use_gpu, double, thaw, temporal, dtype):
    # Switch to train mode
    accumulation_steps=4
    model.train()
    losses = []
    print("Now commencing epoch {}".format(epoch))
    # loss = 0
    optimizer.zero_grad()
    for i, (img,sal,fix,_) in enumerate(train_loader):
        """
        if i == 956 or i == 957:
            #some weird bug happens there
            continue
        """
        # print(type(video))
        img=img['image']
        img,sal,fix=img.cuda(),sal.cuda(),fix.cuda()
        start = datetime.datetime.now().replace(microsecond=0)
        # print("Number of clips for video {} : {}".format(i, len(video)))
        state = None  # Initially no hidden state
        n_iter += i

        # Reset Gradients


        # Squeeze out the video dimension
        # [video_batch, clip_length, channels, height, width]
        # After transpose:
        # [clip_length, video_batch, channels, height, width]

        # clip = Variable(clip.type(dtype).transpose(0, 1))

            # print(clip.size()) #works! torch.Size([5, 1, 1, 360, 640])

            # print(clip[idx].size())

            # Compute output
        saliency_map = model.forward(img)  # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones
        saliency_map = saliency_map.squeeze(1)  # Target is 3 dimensional (grayscale image)

        loss = criterion(saliency_map, sal)

        loss=loss/accumulation_steps

        # Keep score
        losses.append(loss.data*accumulation_steps)

        # Compute gradient
        loss.backward()

        # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
        # nn.utils.clip_grad_norm_(model.parameters(), 10 * clip.size()[0])

        # Update parameters
        if (i+1)%accumulation_steps==0:
            optimizer.step()
            optimizer.zero_grad()

        # # Visualize some of the data
        # if i % 100 == 0 and j == 5:
        #
        #     # writer.add_image('Frame', clip[idx], n_iter)
        #     # writer.add_image('Gtruth', gtruths[idx], n_iter)
        #
        #     post_process_saliency_map = (saliency_map - torch.min(saliency_map)) / (
        #                 torch.max(saliency_map) - torch.min(saliency_map))
        #     utils.save_image(post_process_saliency_map, "./log/smap{}_epoch{}.png".format(i, epoch))
        #
        #     if epoch == 1:
        #         print(saliency_map.max())
        #         print(saliency_map.min())
        #         print(gtruths[idx].max())
        #         print(gtruths[idx].min())
        #         print(post_process_saliency_map.max())
        #         print(post_process_saliency_map.min())
        #         utils.save_image(gtruths[idx], "./log/gt{}.png".format(i))
        #     # writer.add_image('Prediction', prediction, n_iter)

        end = datetime.datetime.now().replace(microsecond=0)
        if i%50==0:
            print('Epoch: {}\t step:{}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch,i,mean(losses),
                                                                                     end - start))

    return (mean(losses), n_iter, optimizer)


def val_img(val_loader, model, criterion, optimizer, epoch, n_iter, use_gpu, double, thaw, temporal, dtype):
    # Switch to train mode
    model.eval()
    summary=Meter()
    print("Now commencing epoch {}".format(epoch))
    # loss = 0
    for i, (img,sal,fix,_) in enumerate(val_loader):
        """
        if i == 956 or i == 957:
            #some weird bug happens there
            continue
        """
        # print(type(video))

        img=img['image']
        img,sal,fix=img.cuda(),sal.cuda(),fix.cuda()
        start = datetime.datetime.now().replace(microsecond=0)
        # print("Number of clips for video {} : {}".format(i, len(video)))
        n_iter += i

            # Compute output
        saliency_map = model.forward(img)  # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones
        saliency_map = saliency_map.squeeze(1)  # Target is 3 dimensional (grayscale image)

        loss = criterion(saliency_map, sal).cpu().detach().numpy()
        saliency_map,fix=saliency_map.cpu().detach().numpy(),fix.cpu().detach().numpy()

        # nss=metric.NSS(saliency_map,fix)



        summary.update(saliency_map,sal,fix,loss)

        # if i%50==0:
        #     nss,loss=summary.get_metrics()
        #     print('Epoch: {}\t val Loss: {}\t nss: {}\t'.format(epoch,loss,nss))
        # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
        # nn.utils.clip_grad_norm_(model.parameters(), 10 * clip.size()[0])

        # # Visualize some of the data
        # if i % 100 == 0 and j == 5:
        #
        #     # writer.add_image('Frame', clip[idx], n_iter)
        #     # writer.add_image('Gtruth', gtruths[idx], n_iter)
        #
        #     post_process_saliency_map = (saliency_map - torch.min(saliency_map)) / (
        #                 torch.max(saliency_map) - torch.min(saliency_map))
        #     utils.save_image(post_process_saliency_map, "./log/smap{}_epoch{}.png".format(i, epoch))
        #
        #     if epoch == 1:
        #         print(saliency_map.max())
        #         print(saliency_map.min())
        #         print(gtruths[idx].max())
        #         print(gtruths[idx].min())
        #         print(post_process_saliency_map.max())
        #         print(post_process_saliency_map.min())
        #         utils.save_image(gtruths[idx], "./log/gt{}.png".format(i))
        #     # writer.add_image('Prediction', prediction, n_iter)

        # end = datetime.datetime.now().replace(microsecond=0)
        # if i==5:
        #     break

    nss, loss = summary.get_metrics()
    return (nss,loss,n_iter, optimizer)


if __name__ == '__main__':
    parser = get_training_parser()
    args = parser.parse_args()
    main(args)

    # utils.save_image(saliency_map.data.cpu(), "test.png")


