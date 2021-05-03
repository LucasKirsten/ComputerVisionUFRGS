import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.init as init


class ASPPModule(nn.Module):

    def __init__(self, features, inner_features=256, out_features=512, dilations=(3, 5, 8)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False))
        self.conv2 = nn.Conv2d(features, inner_features, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv3 = nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0],
                               bias=False)
        self.conv4 = nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1],
                               bias=False)
        self.conv5 = nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2],
                               bias=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            torch.nn.ReLU(True),
            nn.Conv2d(out_features, out_features, kernel_size=1, padding=0, dilation=1),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = torch.nn.functional.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class UNetVgg(torch.nn.Module):
    """
    Combines UNet (VGG based) with the ASPP module for segmentation.
    """

    def __init__(self, nClasses):
        super(UNetVgg, self).__init__()

        vgg16pre = torchvision.models.vgg16(pretrained=True)
        self.vgg0 = torch.nn.Sequential(*list(vgg16pre.features.children())[:4])
        self.vgg1 = torch.nn.Sequential(*list(vgg16pre.features.children())[4:9])
        self.vgg2 = torch.nn.Sequential(*list(vgg16pre.features.children())[9:16])
        self.vgg3 = torch.nn.Sequential(*list(vgg16pre.features.children())[16:23])
        self.vgg4 = torch.nn.Sequential(*list(vgg16pre.features.children())[23:30])

        self.bottom = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            ASPPModule(512)
        )

        self.aux_path = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, nClasses, kernel_size=1, stride=1, padding=0),
        )

        self.smooth0 = torch.nn.Sequential(
            torch.nn.Conv2d(160, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
        )
        self.smooth1 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 96, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(96, 96, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
        )
        self.smooth2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
        )
        self.smooth3 = torch.nn.Sequential(
            torch.nn.Conv2d(768, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
        )
        self.smooth4 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
        )

        self.pass0 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
        )
        self.pass1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
        )

        self.bottom_up = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            torch.nn.ReLU(True)
        )

        self.final = torch.nn.Conv2d(64, nClasses, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = self.vgg0(x)
        feat0 = self.pass0(x)
        x = self.vgg1(x)
        feat1 = self.pass1(x)

        feat2 = self.vgg2(x)
        feat3 = self.vgg3(feat2)
        feat4 = self.vgg4(feat3)
        feat5 = self.bottom(feat4)

        btp = self.bottom_up(feat5)

        _, _, H, W = feat4.size()
        up4 = torch.nn.functional.interpolate(feat5, size=(H, W), mode='bilinear', align_corners=True)
        concat4 = torch.cat([feat4, up4], 1)
        end4 = self.smooth4(concat4)

        _, _, H, W = feat3.size()
        up3 = torch.nn.functional.interpolate(end4, size=(H, W), mode='bilinear', align_corners=True)
        concat3 = torch.cat([feat3, up3], 1)
        end3 = self.smooth3(concat3)

        _, _, H, W = feat2.size()
        up2 = torch.nn.functional.interpolate(end3, size=(H, W), mode='bilinear', align_corners=True)
        concat2 = torch.cat([feat2, up2], 1)
        end2 = self.smooth2(concat2)

        aux_out = self.aux_path(end2)

        _, _, H, W = feat1.size()
        up1 = torch.nn.functional.interpolate(end2, size=(H, W), mode='bilinear', align_corners=True)
        bottom_up = torch.nn.functional.interpolate(btp, size=(H, W), mode='bilinear', align_corners=True)
        concat1 = torch.cat([feat1, up1, bottom_up], 1)
        end1 = self.smooth1(concat1)

        _, _, H, W = feat0.size()
        up0 = torch.nn.functional.interpolate(end1, size=(H, W), mode='bilinear', align_corners=True)
        concat0 = torch.cat([feat0, up0], 1)
        end0 = self.smooth0(concat0)

        aux_out = torch.nn.functional.interpolate(aux_out, size=(H, W), mode='bilinear', align_corners=True)
        return self.final(end0), aux_out
        # return self.final(end0)

    @staticmethod
    def eval_net_with_loss(model, batch, class_weights, device):
        """
        Evaluate network including loss.
        
        Args:
            model (torch.nn.Module): The model.
            inp (torch.tensor): A tensor (float32) of size (batch, 3, H, W)
            gt (torch.tensor): A tensor (long) of size (batch, 1, H, W) with the groud truth (0 to num_classes-1).
            class_weights (list of float): A list with len == num_classes.
            device (torch.device): device to perform computation
            
        Returns:
            out (torch.tensor): Network output.
            loss (torch.tensor): Tensor with the total loss.
                
        """
        images = batch['image'].to(device)
        gt = batch['gt'].to(device)

        weights = torch.from_numpy(np.array(class_weights, dtype=np.float32)).to(device)
        out, aux_out = model(images)

        softmax = torch.nn.functional.log_softmax(out, dim=1)
        softmax_aux = torch.nn.functional.log_softmax(aux_out, dim=1)

        loss = torch.nn.functional.nll_loss(softmax, gt, ignore_index=-1, weight=weights)
        loss_aux = torch.nn.functional.nll_loss(softmax_aux, gt, ignore_index=-1, weight=weights)

        total_loss = loss * 0.6 + loss_aux * 0.4
        return out, total_loss
        # return out, loss

    @staticmethod
    def get_params_by_kind(model, n_base=7):

        base_vgg_bias = []
        base_vgg_weight = []
        core_weight = []
        core_bias = []

        for name, param in model.named_parameters():
            if 'vgg' in name and ('weight' in name or 'bias' in name):
                vgglayer = int(name.split('.')[0][-1])

                if vgglayer <= n_base:
                    if 'bias' in name:
                        print('Adding %s to base vgg bias.' % name)
                        base_vgg_bias.append(param)
                    else:
                        base_vgg_weight.append(param)
                        print('Adding %s to base vgg weight.' % name)
                else:
                    if 'bias' in name:
                        print('Adding %s to core bias.' % name)
                        core_bias.append(param)
                    else:
                        print('Adding %s to core weight.' % name)
                        core_weight.append(param)

            elif 'weight' in name or 'bias' in name:
                if 'bias' in name:
                    print('Adding %s to core bias.' % name)
                    core_bias.append(param)
                else:
                    print('Adding %s to core weight.' % name)
                    core_weight.append(param)

        return base_vgg_weight, base_vgg_bias, core_weight, core_bias

    def init_params(self):
        #        self.init_params_(self.vgg4, False)
        #        self.init_params_(self.vgg3, False)
        #        self.init_params_(self.vgg2, False)
        #        self.init_params_(self.vgg1, False)
        #        self.init_params_(self.vgg0, False)
        self.init_params_(self.smooth4, False)
        self.init_params_(self.smooth3, False)
        self.init_params_(self.smooth2, False)
        self.init_params_(self.smooth1, False)
        self.init_params_(self.smooth0, False)
        self.init_params_(self.bottom, False)
        self.init_params_(self.pass0, False)
        self.init_params_(self.pass1, False)
        #        self.init_params_(self.pbcsa, False)
        self.init_params_(self.bottom_up, False)

        init.kaiming_normal_(self.final.weight, mode='fan_in', nonlinearity='relu')
        if self.final.bias is not None:
            init.constant_(self.final.bias, 0)

    @staticmethod
    def init_params_(model, pre_trained):
        '''Init layer parameters.'''
        for m in model.modules():
            if pre_trained:
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
