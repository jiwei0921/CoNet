import math
import datetime

from torch.autograd import Variable
import torch.nn.functional as F
import torch



running_loss_final = 0
running_loss_pre = 0



def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()

    input = input.transpose(1,2).transpose(2,3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss






class Trainer(object):

    def __init__(self, cuda, model_rgb,model_intergration,model_att, optimizer_rgb,
                 optimizer_inter,optimizer_att,
                 train_loader, max_iter, snapshot, outpath, sshow, size_average=False):
        self.cuda = cuda
        self.model_rgb = model_rgb
        self.model_intergration = model_intergration
        self.model_att = model_att
        self.optim_rgb = optimizer_rgb
        self.optim_inter = optimizer_inter
        self.optim_att = optimizer_att
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.size_average = size_average



    def train_epoch(self):

        for batch_idx, (data, target, depth, edge) in enumerate(self.train_loader):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                data, target, depth, edge = data.cuda(), target.cuda(), depth.cuda(), edge.cuda()
            data, target, depth, edge = Variable(data), Variable(target), Variable(depth), Variable(edge)
            n, c, h, w = data.size()        # batch_size, channels, height, weight


            self.optim_rgb.zero_grad()
            self.optim_inter.zero_grad()
            self.optim_att.zero_grad()

            global running_loss_final
            global running_loss_pre

            low_1, low_2, high_1, high_2, high_3 = self.model_rgb(data)
            Features, _, _, pred_edge1, high_depth, high_sal, pred_depth, pred_sal2 = self.model_intergration(low_1, low_2, high_1, high_2, high_3)

            loss3 = cross_entropy2d(pred_edge1,edge,weight=None,size_average=self.size_average)
            loss4 = F.smooth_l1_loss(high_depth, depth, size_average=self.size_average)
            loss5 = cross_entropy2d(high_sal, target, weight=None, size_average=self.size_average)
            loss6 = F.smooth_l1_loss(pred_depth, depth, size_average=self.size_average)
            loss7 = cross_entropy2d(pred_sal2, target, weight=None, size_average=self.size_average)
            loss_pre = ((loss3+loss5+loss7) + (loss4+loss6)*2.5)/5
            running_loss_pre += loss_pre.item()	    

            loss_pre.backward(retain_graph=True)
            self.optim_inter.step()
            self.optim_rgb.step()

            low_1, low_2, high_1, high_2, high_3 = self.model_rgb(data)
            Features, _, _, Edge, _, _, Depth, Sal = self.model_intergration(low_1, low_2, high_1, high_2, high_3)
            outputs = self.model_att(Features, Edge, Sal, Depth)
            loss_all = cross_entropy2d(outputs, target, weight=None, size_average=self.size_average)
            running_loss_final += loss_all.item()



            if iteration % self.sshow == (self.sshow-1):
                curr_time = str(datetime.datetime.now())[:19]
                print('\n [%s, %3d, %6d,   The training loss of Net: %.3f, and the auxiliary loss: %.3f]' % (curr_time, self.epoch + 1, iteration + 1, running_loss_final / (n * self.sshow),running_loss_pre / (n * self.sshow)))

                running_loss_pre = 0.0
                running_loss_final = 0.0


            if iteration <= 200000:
                if iteration % self.snapshot == (self.snapshot-1):
                    savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (snapshot: %d)' % (iteration+1))
                
                    savename_focal = ('%s/integrate_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_intergration.state_dict(), savename_focal)
                    print('save: (snapshot_integrate: %d)' % (iteration+1))

                    savename_clstm = ('%s/att_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                    torch.save(self.model_att.state_dict(), savename_clstm)
                    print('save: (snapshot_att: %d)' % (iteration+1))

            else:
                if iteration % 20000 == (20000 - 1):
                    savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (snapshot: %d)' % (iteration + 1))

                    savename_focal = ('%s/integrate_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_intergration.state_dict(), savename_focal)
                    print('save: (snapshot_integrate: %d)' % (iteration + 1))

                    savename_clstm = ('%s/att_snapshot_iter_%d.pth' % (self.outpath, iteration + 1))
                    torch.save(self.model_att.state_dict(), savename_clstm)
                    print('save: (snapshot_att: %d)' % (iteration + 1))



            if (iteration+1) == self.max_iter:
                savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_rgb.state_dict(), savename)
                print('save: (snapshot: %d)' % (iteration+1))

                savename_focal = ('%s/integrate_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_intergration.state_dict(), savename_focal)
                print('save: (snapshot_integrate: %d)' % (iteration+1))

                savename_clstm = ('%s/att_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_att.state_dict(), savename_clstm)
                print('save: (snapshot_att: %d)' % (iteration+1))


            loss_all.backward()
            self.optim_att.step()
            self.optim_inter.step()
            self.optim_rgb.step()

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
