import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer
from collections import OrderedDict
import segmentation_models_pytorch as smp
from models_unet_cnntransf.build_model import build_segmodel
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import os


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, initmodel):
    print(1, torch.cuda.is_available())
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    # model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    if config['use_smp']:
        model = getattr(smp, config["arch"]['smp']["decoder_name"])(
                                                                    encoder_name=config["arch"]['smp']['encoder_name'],
                                                                    encoder_weights='imagenet',
                                                                    classes=config["class_num"],
                                                                    #encoder_depth=5,
                                                                    #encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36), in_channels=3, upsampling=4, aux_params=None,
                                                                    activation=None
                                                                    )
        print('build smp model done')
    else:
        model = build_segmodel(modelname=config['arch']['type'], 
                               imgsize=config['train_loader']['args']['crop_size'], 
                               numclasses=config["class_num"])
        print('build unet series model done')
    # print(f'\n{model}\n')

    """
    # init model
    if initmodel:
        initmodel = initmodel + '/best_model.pth'
        state_dict = torch.load(initmodel)['state_dict']
        new_state_dict = OrderedDict()
        if config['use_smp']:
            for k, v in state_dict.items():
                if 'segmentation_head' not in k:
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v 
        else:
            for k, v in state_dict.items():
                if 'swin_unet.output' not in k and 'segmentation_head' not in k and 'final.' not in k and 'Conv.' not in k:
                    name = k[7:]
                    new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print('load segmodel and init done')
        print('load pretrained model from ', initmodel)

    """
    if initmodel:
        initmodel = initmodel
        state_dict = torch.load(initmodel, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('segmentation_head'):
                del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(initmodel))
        print("missing keys:", msg.missing_keys)

    print(2, torch.cuda.is_available())

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()

if __name__=='__main__':

    print(0, torch.cuda.is_available())

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default='all', type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-m', '--initmodel', default=None, type=str,
                           help='model path for init model')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume, args.initmodel)