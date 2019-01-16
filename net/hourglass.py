import torch
from torch import nn
from torch.nn import functional as F
from net.layer_utils import  *
from libs import TopPool, LeftPool, RightPool,BottomPool

class Hourglass_model(nn.Module):
    def __init__(self, n, dims, num_layers, **kwargs):
        super(Hourglass_model, self).__init__()
        self.n   = n
        curr_mod = num_layers[0]
        next_mod = num_layers[1]
        curr_dim = dims[0]
        next_dim = dims[1]


        self.up1  = self.make_layer(
            inp_dim=curr_dim, out_dim=curr_dim, kenel=3,  num_layers=curr_mod, layer=Residual,
            **kwargs)

        self.max1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = self.make_layer(
            inp_dim=curr_dim, out_dim=next_dim, kenel=3, num_layers=curr_mod, layer=Residual,
            **kwargs)

        self.low2 = Hourglass_model(
            n - 1, dims[1:], num_layers[1:],
            **kwargs
        ) if self.n > 1 else \
        self.make_layer(
            inp_dim=next_dim, out_dim=next_dim,kenel=3, num_layers= next_mod,
            layer=Residual, **kwargs
        )
        self.low3 = self.make_layer_revert(
            inp_dim=next_dim, out_dim=curr_dim, kenel=3, num_layers= curr_mod,
            layer=Residual, **kwargs
        )
        #self.up2  = nn.Upsample(scale_factor=2)

        self.merge = MergeUp()

    def make_layer(self,inp_dim, out_dim, kenel, num_layers, layer=ConBnRelu, **kwargs):
        layers = [layer(inp_dim, out_dim,kenel, **kwargs)]
        for _ in range(1, num_layers):
            layers.append(layer(out_dim, out_dim, kenel, **kwargs))
        return nn.Sequential(*layers)

    def make_layer_revert(self,inp_dim, out_dim, kenel, num_layers, layer=ConBnRelu, **kwargs):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(layer(inp_dim, inp_dim,kenel, **kwargs))
        layers.append(layer(inp_dim, out_dim, kenel, **kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = F.interpolate(low3, scale_factor=2)

        return self.merge(up1, up2)



class CornerNet(nn.Module):
    def __init__(self):
        super(CornerNet,self).__init__()
        self.n = 5
        self.dims = [256, 256, 384, 384, 384, 512]
        self.num_layers = [2, 2, 2, 2, 2, 4]
        self.n_stack = 2
        self.conv_dim = 256
        self.curr_dim = self.dims[0]
        self.out_dim = 20
        self.pre = nn.Sequential(
            ConBnRelu(inp_dim=3, out_dim=128,kenel=7 ,stride=2),
            Residual(inp_dim=128, out_dim=256, kernel=3, stride=2)
        )
        self.hourglass = nn.ModuleList([
             Hourglass_model(
                n = self.n, dims = self.dims, num_layers=self.num_layers
            ) for _ in range(self.n_stack)
        ])

        self.cnvs = nn.ModuleList([
            ConBnRelu(inp_dim=self.curr_dim, out_dim=self.conv_dim, kenel=3),
            ConBnRelu(inp_dim=self.conv_dim, out_dim=self.conv_dim, kenel=3)
        ])

        self.top_left_layer = nn.ModuleList([
            CornerPool(dim=self.conv_dim, pool1=TopPool, pool2=LeftPool),
            CornerPool(dim=self.conv_dim, pool1=TopPool, pool2=LeftPool)
        ])

        self.bottom_right_layer = nn.ModuleList([
            CornerPool(dim=self.conv_dim, pool1=BottomPool, pool2=RightPool),
            CornerPool(dim=self.conv_dim, pool1=BottomPool, pool2=RightPool)
        ])

        self.top_left_heats = nn.ModuleList([
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, self.out_dim, (1, 1))
            ),
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, self.out_dim, (1, 1))
            )
        ])

        self.bottom_right_heats = nn.ModuleList([
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, self.out_dim, (1, 1))
            ),
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, self.out_dim, (1, 1))
            )
        ])

        self.top_left_tags = nn.ModuleList([
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 1, (1, 1))
            ),
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 1, (1, 1))
            )
        ])

        self.bottom_right_tags = nn.ModuleList([
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 1, (1, 1))
            ),
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 1, (1, 1))
            )
        ])

        for tl_heat, br_heat in zip(self.top_left_heats, self.bottom_right_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)

        self.inters1 = nn.ModuleList([
            Residual(self.curr_dim, self.curr_dim, kernel=3)
        ])

        self.inters2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.curr_dim, self.curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(self.curr_dim)
            )
        ])

        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.conv_dim, self.curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(self.curr_dim)
            )
        ])

        self.top_left_regrs = nn.ModuleList([
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 2, (1, 1))
            ),
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 2, (1, 1))
            )
        ])
        self.bottom_right_regrs = nn.ModuleList([
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 2, (1, 1))
            ),
            nn.Sequential(
                ConBnRelu(self.conv_dim, self.curr_dim, kenel=3, with_bn=False),
                nn.Conv2d(self.curr_dim, 2, (1, 1))
            )
        ])

        self.relu = nn.ReLU(inplace=True)





    def forward(self, ig, tl_inds, br_inds):
        inter = self.pre(ig)
        outs = []

        layers = zip(
            self.hourglass, self.cnvs,
            self.top_left_layer, self.bottom_right_layer,
            self.top_left_heats, self.bottom_right_heats,
            self.top_left_tags, self.bottom_right_tags,
            self.top_left_regrs, self.bottom_right_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            tl_cnv_, br_cnv_ = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_ = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]


            kp = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)



            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)


            tl_tag, br_tag = tl_tag_(tl_cnv), br_tag_(br_cnv)

            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            tl_tag = tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag = tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = tranpose_and_gather_feat(br_regr, br_inds)

            outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < 1:
                inter = self.inters2[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters1[ind](inter)
        return outs





if __name__ == '__main__':
    from data_set.data_gen import get_batch

    gen = get_batch(batch_size=1, class_name='voc', max_detect=100)
    md = CornerNet()
    d = next(gen)
    ig = torch.from_numpy(d[0]).float()
    tl_inds = torch.from_numpy(d[1])
    br_inds = torch.from_numpy(d[2])
    print(len(md(ig, tl_inds, br_inds)))



