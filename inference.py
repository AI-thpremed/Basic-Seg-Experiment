import os
import json
import argparse
import torch
import PIL
import numpy as np
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from models_unet_cnntransf.build_model import build_segmodel
from PIL import ImageOps


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--modelpath', default='best_model.pth', type=str,
                        help='The config used to train the model')
    parser.add_argument('-i', '--images', default='validation/', type=str,
                        help='The config used to train the model')
    parser.add_argument('-s', '--imgsize', default=512, type=int,
                        help='The config used to train the model')
    parser.add_argument('-o', '--output', default='unet/', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mode', '--imgmode', default='.jpeg', type=str,
                        help='The config used to train the model')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    transform_val = transforms.Compose([
                                        transforms.Resize((args.imgsize, args.imgsize)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])

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
        
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.modelpath, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    for imgname in tqdm(os.listdir(args.images)):
        i_imgpath = args.images + imgname
        # o_imgpath = args.output + "Res_"+imgname[:-4] + '.' + args.imgmode
        o_imgpath = args.output + "Res_"+imgname[:-4] +  args.imgmode
        
        img = Image.open(i_imgpath).convert('RGB')
        w, h = img.size
        input = transform_val(img).unsqueeze(0)

        prediction = model(input.to(device))
        prediction = prediction.squeeze(0).cpu().detach().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

        # mask_0 = (prediction == 0) * 0
        # mask_1 = (prediction == 1) * 128
        # mask_2 = (prediction == 2) * 255
        # mask = mask_0 + mask_1 + mask_2
        """#yuanlaide heibaihuimsk
        mask_0 = (prediction == 0) * 0
        mask_1 = (prediction == 1) * 96
        mask_2 = (prediction == 2) * 178
        mask_3 = (prediction == 3) * 255
        mask = mask_0 + mask_1 + mask_2 + mask_3


        pred_mask = Image.fromarray(np.uint8(mask))
        pred_mask = pred_mask.resize((w, h), resample=PIL.Image.Resampling.NEAREST).convert('L')

        pred_mask.save(o_imgpath)
        """
        # 创建一个空的3通道数组，用于存储颜色
        mask_rgb = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

        # mask_0: 黑色，即 [0, 0, 0]
        mask_0 = (prediction == 0)
        mask_rgb[mask_0] = [0, 0, 0]
    
        # mask_1: 红色，即 [255, 0, 0]
        mask_1 = (prediction == 1)
        mask_rgb[mask_1] = [255, 0, 0]
    
        # mask_2: 蓝色，即 [0, 0, 255]
        mask_2 = (prediction == 2)
        mask_rgb[mask_2] = [0,  0,255]
    
        # mask_3: 绿色，即 [0, 255, 0]
        mask_3 = (prediction == 3)
        mask_rgb[mask_3] = [0,  255,0]


        # mask_4 = (prediction == 4)
        # mask_rgb[mask_4] = [255,255,0]

        
        # mask_5 = (prediction == 5)
        # mask_rgb[mask_5] = [255,  0,255]


     # if label == "SRF":
            #     color = (255, 0, 0)  # 蓝色
            # elif label == "IRF":
            #     color = (0, 255, 0)  # 绿色
            # elif label == "PED":
            #     color = (0, 0, 255)  # 红色
            # elif label == "SHRM":
            #     color = (255, 255, 0)  # 黄色
            # elif label == "IS/OS":
            #     color = (255, 0, 255)  # 品红色


        pred_mask = Image.fromarray(mask_rgb)
        pred_mask = pred_mask.resize((w, h), resample=Image.NEAREST)

        pred_mask.save(o_imgpath)
  
if __name__ == '__main__':
    main()

#结果要测出来，比如说我这个域适应，直接在整个数据集上跑出结果来，怎么找到对应的mask。 