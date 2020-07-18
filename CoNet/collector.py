import torch
import torch.nn as nn
import torch.nn.functional as F



class TriAtt(nn.Module):
    def __init__(self):
        super(TriAtt, self).__init__()

        self.conv_concat = nn.Conv2d(2*2, 2, 1, padding=0)
        self.predict = nn.Conv2d(128, 2, 1, padding=0)


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self,Features, Edge, Sal, Depth):

        # -------------------- Knowledge Collector ------------------- #
        Feature_d = torch.mul(Features, Depth)
        Feature_d = Features + Feature_d

        Att = torch.cat([Edge,Sal],dim=1)
        Att = self.conv_concat(Att)
        Att = F.softmax(Att,dim=1)[:,1:,:]

        Feature_dcs = torch.mul(Feature_d, Att)
        Feature_all = Feature_dcs + Feature_d

        outputs = self.predict(Feature_all)

        return outputs

