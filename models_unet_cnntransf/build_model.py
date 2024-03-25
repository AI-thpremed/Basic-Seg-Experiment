import torch
import numpy as np
# from models_unet_cnntransf.swin_unet.vision_transformer import SwinUnet
# from models_unet_cnntransf.smeswin_unet.vision_transformer import SMESwinUnet
# from models_unet_cnntransf.trans_unet.vit_seg_modeling import VisionTransformer as ViT_seg
# from models_unet_cnntransf.trans_unet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from models_unet_cnntransf.medt.axialnet import MedT, axialunet, gated, logo
# from models_unet_cnntransf.unext.unext_models import UNext, UNext_S
from models_unet_cnntransf.unet_series.unet_series_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
# from models_unet_cnntransf.unet_2022.model_unet2022.UNet2022 import unet2022
# from models_unet_cnntransf.unet_2022.model_unet2022.UNet2022 import init_model
# from models_unet_cnntransf.unet_2022.utilities.nd_softmax import softmax_helper
# from models_unet_cnntransf.scaleformer.ScaleFormer import ScaleFormer
# from models_unet_cnntransf.isnet.isnet import ISNetDIS_1, ISNetDIS_2
# from models_unet_cnntransf.u2net.u2net import U2NET, U2NETP, U2NET_2, U2NETP_2
# from models_unet_cnntransf.cenet.cenet import CE_Net_
from models_unet_cnntransf.nextunet.beifen.unext_models import UNetConvNeXtV2Tiny,UNetConvNeXtV2Pico,UNetConvNeXtV2Base
# from models_unet_cnntransf.hformer.H2Former import res34_swin_MS
# from models_unet_cnntransf.model.DconnNet import DconnNet
# from models_unet_cnntransf.mstgan.MsTGANet import MsTGANet
#from models_unet_cnntransf.nextunet.unext_models import UNetConvNeXtV2Tiny,UNetConvNeXtV2Pico,UNetConvNeXtV2Base

pretrained_models_dict = {
    'swinunet': '/vepfs/niuzhiyuan/pretrainmodel/swin_tiny_patch4_window7_224_22k.pth',
    'unet2022': '/data/niuzy/python20233/pretrainmodels/convnext_t_3393.model',
    'vitseg': '/vepfs/niuzhiyuan/pretrainmodel/imagenet21k_R50+ViT-B_16.npz',
    'isnet_1': '/data/chencc/pretrained_models/image_seg/isnet.pth',
    'isnet_2': '/data/chencc/pretrained_models/image_seg/isnet.pth',
    'u2net': '/data/chencc/pretrained_models/image_seg/u2net.pth',
    'u2netp': '/data/chencc/pretrained_models/image_seg/u2netp.pth',
    #'nextunet':'/data/niuzy/python20233/pretrainmodels/convnextv2_tiny_1k_224_fcmae.pt',
    # 'nextunet':'/vepfs/niuzhiyuan/pretrainmodel/convnextv2/convnextv2_tiny_22k_384_ema.pt',

    'nextunet':'/vepfs/niuzhiyuan/pretrainmodel/convnextv2/convnextv2_tiny_22k_384_ema.pt',

    #'nextunet':'/vepfs/niuzhiyuan/pretrainmodel/convnextv2/convnextv2_tiny_22k_224_ema.pt',
    'nextunetb':'/vepfs/niuzhiyuan/pretrainmodel/convnextv2/convnextv2_base_22k_384_ema.pt',
    'nextunetp':'/vepfs/niuzhiyuan/pretrainmodel/convnextv2/convnextv2_pico_1k_224_ema.pt'

    }

def build_segmodel(modelname='swinunet', numclasses=4, imgsize=512):
    if modelname == 'swinunet' :
        model = SwinUnet(img_size=imgsize, num_classes=numclasses, window_size=7)
        model.load_from(pretrained_model_path=pretrained_models_dict[modelname])
        print('init model from ', pretrained_models_dict[modelname])
    elif modelname == 'unet2022':
        model = unet2022(num_classes=numclasses, window_size=[8,8,16,8], img_size=[imgsize,imgsize])
        model = init_model(model, torch.load(pretrained_models_dict[modelname]))
        print('init model from ', pretrained_models_dict[modelname])
    
    elif modelname == 'nextunet':
        model = UNetConvNeXtV2Tiny(in_chans=3, num_classes=numclasses)
        print('model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )

        model.load_from(pretrained_model_path=pretrained_models_dict[modelname])
        print('init model from ', pretrained_models_dict[modelname])

    elif modelname == 'nextunetb':
        model = UNetConvNeXtV2Base(in_chans=3, num_classes=numclasses)
        model.load_from(pretrained_model_path=pretrained_models_dict[modelname])
        print('init model from ', pretrained_models_dict[modelname])

    elif modelname == 'nextunetp':
        model = UNetConvNeXtV2Pico(in_chans=3, num_classes=numclasses)
        model.load_from(pretrained_model_path=pretrained_models_dict[modelname])
        print('init model from ', pretrained_models_dict[modelname])

    elif modelname == 'dconn':
        model = DconnNet(num_class=numclasses).cuda()
    
    elif modelname == 'mst':
        model = MsTGANet(in_channels=3, num_classes=numclasses)



    elif modelname == 'vitseg':
        vit_name = 'R50-ViT-B_16'
        vit_patches_size = 16
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = numclasses
        config_vit.n_skip = 3
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(imgsize / vit_patches_size), int(imgsize / vit_patches_size))
        model = ViT_seg(config_vit, img_size=imgsize, num_classes=config_vit.n_classes)
        model.load_from(weights=np.load(pretrained_models_dict[modelname]))
        print('init model from ', pretrained_models_dict[modelname])
        print('model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )

    elif modelname == 'isnet_1':
        del_weight = ['side1.weight', 'side1.bias', 'side2.weight', 'side2.bias', 'side3.weight', 'side3.bias',
                      'side4.weight', 'side4.bias', 'side5.weight', 'side5.bias', 'side6.weight', 'side6.bias']
        model = ISNetDIS_1(out_ch=numclasses)
        checkpoint = torch.load(pretrained_models_dict[modelname])
        for name in del_weight:
            del checkpoint[name]
        model.load_state_dict(checkpoint, strict=False)
        print('init model from ', pretrained_models_dict[modelname])
    elif modelname == 'isnet_2':
        del_weight = ['side1.weight', 'side1.bias', 'side2.weight', 'side2.bias', 'side3.weight', 'side3.bias',
                      'side4.weight', 'side4.bias', 'side5.weight', 'side5.bias', 'side6.weight', 'side6.bias']
        model = ISNetDIS_2(out_ch=numclasses)
        checkpoint = torch.load(pretrained_models_dict[modelname])
        for name in del_weight:
            del checkpoint[name]
        model.load_state_dict(checkpoint, strict=False)
        print('init model from ', pretrained_models_dict[modelname])
    elif modelname == 'u2net':
        del_weight = ['side1.weight', 'side1.bias', 'side2.weight', 'side2.bias', 'side3.weight', 'side3.bias',
                      'side4.weight', 'side4.bias', 'side5.weight', 'side5.bias', 'side6.weight', 'side6.bias',
                      'outconv.weight', 'outconv.bias']
        model = U2NET(out_ch=numclasses)
        checkpoint = torch.load(pretrained_models_dict[modelname])
        for name in del_weight:
            del checkpoint[name]
        model.load_state_dict(checkpoint, strict=False)
        print('init model from ', pretrained_models_dict[modelname])
    elif modelname == 'u2net_2':
        del_weight = ['side1.weight', 'side1.bias', 'side2.weight', 'side2.bias', 'side3.weight', 'side3.bias',
                      'side4.weight', 'side4.bias', 'side5.weight', 'side5.bias', 'side6.weight', 'side6.bias',
                      'outconv.weight', 'outconv.bias']
        model = U2NET_2(out_ch=numclasses)
        checkpoint = torch.load(pretrained_models_dict[modelname])
        for name in del_weight:
            del checkpoint[name]
        model.load_state_dict(checkpoint, strict=False)
        print('init model from ', pretrained_models_dict[modelname])
    elif modelname == 'u2netp':
        del_weight = ['side1.weight', 'side1.bias', 'side2.weight', 'side2.bias', 'side3.weight', 'side3.bias',
                      'side4.weight', 'side4.bias', 'side5.weight', 'side5.bias', 'side6.weight', 'side6.bias',
                      'outconv.weight', 'outconv.bias']
        model = U2NETP(out_ch=numclasses)
        checkpoint = torch.load(pretrained_models_dict[modelname])
        for name in del_weight:
            del checkpoint[name]
        model.load_state_dict(checkpoint, strict=False)
        print('init model from ', pretrained_models_dict[modelname])
    elif modelname == 'u2netp_2':
        del_weight = ['side1.weight', 'side1.bias', 'side2.weight', 'side2.bias', 'side3.weight', 'side3.bias',
                      'side4.weight', 'side4.bias', 'side5.weight', 'side5.bias', 'side6.weight', 'side6.bias',
                      'outconv.weight', 'outconv.bias']
        model = U2NETP_2(out_ch=numclasses)
        checkpoint = torch.load(pretrained_models_dict[modelname])
        for name in del_weight:
            del checkpoint[name]
        model.load_state_dict(checkpoint, strict=False)
        print('init model from ', pretrained_models_dict[modelname])

    elif modelname == 'cenet':
        model = CE_Net_(num_classes=numclasses)
    elif modelname == 'medt':
        model = MedT(img_size=imgsize, num_classes=numclasses)
    elif modelname == 'scaleformer':
        model = ScaleFormer(n_classes=numclasses)
    elif modelname == 'unet':
        model = U_Net(in_ch=3, out_ch=numclasses)
    elif modelname == 'attunet':
        model = AttU_Net(in_ch=3, out_ch=numclasses)
    elif modelname == 'unetpp':
        model = NestedUNet(in_ch=3, out_ch=numclasses)
    elif modelname == 'unext':
        model = UNext(num_classes=numclasses, input_channels=3, deep_supervision=False, img_size=imgsize)
    elif modelname == 'h2former':
        model = res34_swin_MS(512, 4)  #def res34_swin_MS(image_size, num_class) :    return Res34_Swin_MS(image_size, BasicBlock, [3, 4, 6, 3],num_classes = num_class)
        print('model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
        model_dict = model.state_dict()
        pre_dict = torch.load('/vepfs/niuzhiyuan/pretrainmodel/resnet34-333f7ec4.pth') 
        matched_dict = {k: v for k, v in pre_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
        print('matched keys:', len(matched_dict))
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

    else:
        print('The model name is incorrect')

    return model
