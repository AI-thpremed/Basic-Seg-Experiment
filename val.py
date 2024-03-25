import os
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
import segmentation_models_pytorch as smp
from models_unet_cnntransf.build_model import build_segmodel


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, modelpath):
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
                                                                    activation=None
                                                                    )
        print('build smp model done')
    else:
        model = build_segmodel(modelname=config['arch']['type'], 
                               imgsize=config['train_loader']['args']['crop_size'], 
                               numclasses=config["class_num"])
        print('build unet series model done')

    # load trained model
    state_dict = torch.load(modelpath)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
    # load params
    model.load_state_dict(new_state_dict)
    print('load trained model from ', modelpath)

    # print(f'\n{model}\n')

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

    trainer.val()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='./configs/config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    _config = args.config + '/config.json'
    _modelpath = args.config + '/best_model.pth'

    config = json.load(open(_config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume, _modelpath)