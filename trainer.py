import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter, dice_score ,avd_score
from tqdm import tqdm
from medpy import metric
from medpy.metric.binary import hd

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.loss_fn(input, target)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        return torch.mean((output - target) ** 2)


class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        self.MSELoss = MSELoss()  # 添加这行代码
        self.CrossEntropyLoss = CrossEntropyLoss()  # 添加这行代码
        self.Cross = nn.CrossEntropyLoss()
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        

        # self.device='cuda:6'
        if self.device ==  torch.device('cpu'): prefetch = False

        # print(self.device)

        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

        print(self.device)

    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        # print('lr:', self.optimizer.param_groups[-1]['lr'])
        #class_dice_per_image = {i: [] for i in range(1, self.num_classes)}

        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            #data, target = data.to(self.device), target.to(self.device)
            self.lr_scheduler.step(epoch=epoch-1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.config['arch']['type'][:3] == 'PSP':
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes 
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                # assert output.size()[2:] == target.size()[1:]
                # assert output.size()[1] == self.num_classes 
                # loss = self.loss(output, target)
                # #loss += self.CrossEntropyLoss(output[1], target) * 0.1
                # # loss += self.MSELoss(output[1], output[2]) * 0.1
                # # loss += self.MSELoss(output[3], output[4]) * 0.1
                # # loss += self.MSELoss(output[5], output[6]) * 0.1
                # # loss += self.CrossEntropyLoss(output[7], output[8]) * 0.05
                # # loss += self.CrossEntropyLoss(output[9], output[10]) * 0.05
                # # loss += self.CrossEntropyLoss(output[11], output[12]) * 0.05
                # # loss += self.CrossEntropyLoss(output[13], output[12]) * 0.05
                # output = output

                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes 
                loss = self.loss(output, target)
                # loss += self.loss(output[1],torch.argmax(output[2], dim=1)) * 0
                # loss += self.loss(output[3],torch.argmax(output[4], dim=1)) * 0
                # loss += self.loss(output[5],torch.argmax(output[6], dim=1)) * 0
                # loss += self.loss(output[7],torch.argmax(output[8], dim=1)) * 0

                #####loss += self.CrossEntropyLoss(output[1], target) * 0.1
                # ####loss += self.MSELoss(output[1], output[2]) * 0.1
                # ####loss += self.MSELoss(output[3], output[4]) * 0.1
                # ####loss += self.MSELoss(output[5], output[6]) * 0.1
                # loss += self.CrossEntropyLoss(output[7], output[8]) * 0.05
                # loss += self.CrossEntropyLoss(output[9], output[10]) * 0.05
                # loss += self.CrossEntropyLoss(output[11], output[12]) * 0.05
                # loss += self.CrossEntropyLoss(output[13], output[12]) * 0.05
                """ """
                # #loss += self.Cross(output[1], torch.argmax(output[2], dim=1)) * 0.05
                # loss += self.Cross(output[3], torch.argmax(output[4], dim=1)) * 0.01
                # loss += self.Cross(output[5], torch.argmax(output[6], dim=1)) * 0.01
                # loss += self.Cross(output[7], torch.argmax(output[8], dim=1)) * 0.01
                # #loss += self.Cross(output[9], torch.argmax(output[10], dim=1)) * 0.05
                # loss += self.Cross(output[11], torch.argmax(output[12], dim=1)) * 0.01
                # loss += self.Cross(output[13], torch.argmax(output[14], dim=1)) * 0.01
                # #loss += self.Cross(output[15], torch.argmax(output[16], dim=1)) * 0.05
                # loss += self.Cross(output[17], torch.argmax(output[18], dim=1)) * 0.01
                # loss += self.Cross(output[19], torch.argmax(output[20], dim=1)) * 0.01

                #output = output[0]


            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
            mDice, _ = self._get_seg_metrics_2().values()#
            #seg_metrics_2 = self._get_seg_metrics_2()
            # Record class-wise Dice for each image
            #for cls, dice in enumerate(class_dice.values()):
            #    class_dice_per_image[cls + 1].append(dice)

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.4f} mIoU {:.5f} mDice {:.5f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average, 
                                                pixAcc, mIoU, mDice,
                                                self.batch_time.average, self.data_time.average))

       # Compute average class-wise Dice for all images
        #avg_class_dice = {cls: np.nanmean(dices) for cls, dices in class_dice_per_image.items()}
        #avg_all_dice = np.nanmean(list(avg_class_dice.values()))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        seg_metrics_2 = self._get_seg_metrics_2()
        for k, v in list(seg_metrics.items())[:-1]: 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics,
                **seg_metrics_2,
                #**avg_class_dice,
                #**avg_all_dice
                }

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def dice_coef(self, predictions, labels, smooth=1):
        dice = {}
        for i in range(labels.shape[-1]):
            label_1D = labels[:, :, i].flatten()
            pred_1D  = predictions[:, :, i].flatten()

            intersection = torch.sum(label_1D * pred_1D)

            try:
                dice_coff = (2. * intersection + smooth) / (torch.sum(label_1D) + torch.sum(pred_1D) + smooth)
                dice[self.layers_names[i]] = float(dice_coff)
            except:
                dice_coff = 0
                dice[self.layers_names[i]] = dice_coff

        return dice

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)

        #class_dice_per_image = {i: [] for i in range(1, self.num_classes)}

        with torch.no_grad():
            val_visual = []

            dice_scores_per_class = [0.0] * self.num_classes
            dice_per_class = [0.0] * self.num_classes
            avd_scores_per_class = [0.0] * self.num_classes
            avd_scores_per_class1 = [0.0] * self.num_classes
            iou_scores_per_class = [0.0] * self.num_classes
            #hd_scores_per_class = [0.0] * self.num_classes
            #acc_scores_per_class = [0.0] * self.num_classes
            tp_counts = [0.0] * self.num_classes
            fp_counts = [0.0] * self.num_classes
            tn_counts = [0.0] * self.num_classes
            fn_counts = [0.0] * self.num_classes
            num_images_per_class = [0] * self.num_classes
            #allimagesnum = [0] * self.num_classes

            for batch_idx, (data, target) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                #output = output[0] ###在总这里设置一下
                """这里改成不密集损失函数的样式"""
                ######output = output[0] ###在总这里设置一下
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                 # Compute dice scores for each class
                for i in range(data.shape[0]):
                    for j in range(self.num_classes):
                        # predictions = output[i].unsqueeze(0)
                        # labels = target[i].unsqueeze(0)
                        # print("Shape of predictions:", predictions.shape)
                        # print("Shape of labels:", labels.shape)
                        # dice_scores = self.dice_coef(predictions, labels)
                        # dice_per_class[j] += dice_scores[j]
                        # allimagesnum += 1

                        class_target = (target[i] == j).float()
                        if class_target.sum() == 0:
                            continue
                        class_output = (output[i].max(0)[1] == j).float()  #尝试改一下这个地方
                        #class_output = (output[i].argmax(dim=0) == j).float()

                        dice = dice_score(class_output, class_target)
                        #hausdorff = hd(class_output.cpu().numpy(), class_target.cpu().numpy())
                        #if np.any(class_output.cpu().numpy()) and np.any(class_target.cpu().numpy()):
                        #    hausdorff = hd(class_output.cpu().numpy(), class_target.cpu().numpy())
                        #else:
                        #    hausdorff = 0  # 如果没有匹配的类别，将Hausdorff距离设置为NaN
                            #hausdorff = float('nan')  # 如果没有匹配的类别，将Hausdorff距离设置为NaN
                        #avd = avd_score(class_output, class_target)
                        intersection = (class_output * class_target).sum()
                        union = class_output.sum() + class_target.sum() - intersection
                        iou = intersection / union if union != 0 else 0.0
                        iou_scores_per_class[j] += iou

                        avd = abs(class_output.sum() - class_target.sum()) / class_target.sum()
                        avd1 = abs(class_output.sum() - class_target.sum())
                        #acc = torch.sum(class_output * class_target) / torch.sum(class_target)                        
                        dice_scores_per_class[j] += dice
                        avd_scores_per_class[j] += avd
                        avd_scores_per_class1[j] += avd1
                        tp_counts[j] += torch.sum(class_output * class_target).item()
                        fp_counts[j] += torch.sum(class_output * (1 - class_target)).item()
                        tn_counts[j] += torch.sum((1 - class_output) * (1 - class_target)).item()
                        fn_counts[j] += torch.sum((1 - class_output) * class_target).item()
                        #acc_scores_per_class[j] += acc.item()
                        #hd_scores_per_class[j] += hausdorff
                        num_images_per_class[j] += 1
            
                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)   

                #seg_metrics_2 = self._get_seg_metrics_2()
                #_, class_dice = seg_metrics_2.values()
                #for cls, dice in enumerate(class_dice.values()):
                #    class_dice_per_image[cls + 1].append(dice)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy() 
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                mDice, _ = self._get_seg_metrics_2().values()
                #seg_metrics_2 = self._get_seg_metrics_2()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.4f}, Mean IoU: {:.5f}, Mean Dice: {:.5f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU, mDice))
            # for j in range(self.num_classes):
            #     dice_per_class[j] /= allimagesnum[j]
            
            # Compute average dice scores for each class
            avg_dice_scores_per_class = [dice_scores_per_class[i] / num_images_per_class[i] if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            avg_avd_scores_per_class = [avd_scores_per_class[i] / num_images_per_class[i] if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            avg_acc_scores_per_class = [(tp_counts[i] / (tp_counts[i] + fn_counts[i]) + tn_counts[i] / (tn_counts[i] + fp_counts[i])) / 2 if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            avd_scores_per_class1 = [avd_scores_per_class1[i] / num_images_per_class[i] if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            avd_acc_scores_per_class1 = [((tp_counts[i] + tn_counts[i])/ (tp_counts[i] + fn_counts[i] + tn_counts[i] + fp_counts[i]))  if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            #avg_acc_scores_per_class = [acc_scores_per_class[i] / num_images_per_class[i] if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            #avg_dice_scores= (avg_dice_scores_per_class[1] + avg_dice_scores_per_class[2] + avg_dice_scores_per_class[3])/3
            avg_dice_scores = sum(avg_dice_scores_per_class[1:])/(self.num_classes - 1)
            avg_avd_scores = sum(avg_avd_scores_per_class[1:])/(self.num_classes - 1)
            avg_acc_scores = sum(avg_acc_scores_per_class[1:]) / (self.num_classes - 1)
            avg_avd_scores1 = sum(avd_scores_per_class1[1:])/(self.num_classes - 1)
            avg_acc_scores1 = sum(avd_acc_scores_per_class1[1:]) / (self.num_classes - 1)
            #avg_hd_scores_per_class = [hd_scores_per_class[i] / num_images_per_class[i] if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            #avg_hd_scores = sum(avg_hd_scores_per_class[1:]) / (self.num_classes - 1)
            #avg_acc_scores = sum(avg_acc_scores_per_class[1:])/(self.num_classes - 1)     
            avg_iou_scores_per_class = [iou_scores_per_class[i] / num_images_per_class[i] if num_images_per_class[i] != 0 else 0.0 for i in range(self.num_classes)]
            avg_iou_scores = sum(avg_iou_scores_per_class[1:])/(self.num_classes - 1)

            
            # Log the average dice scores
            for i in range(self.num_classes):
                self.logger.info(f"Class {i} average dice score: {avg_dice_scores_per_class[i]}")
                self.logger.info(f"Class {i} average avd score: {avg_avd_scores_per_class[i]}")
                #self.logger.info(f"Class {i} average Hausdorff distance score: {avg_hd_scores_per_class[i]}")
            self.logger.info(f"average_dice_score: {avg_dice_scores}")
            self.logger.info(f"Average avd score: {avg_avd_scores}")
            #self.logger.info(f"Average Hausdorff distance score: {avg_hd_scores}")



            # WRTING & VISUALIZING THE MASKS
            
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            seg_metrics_2 = self._get_seg_metrics_2()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            

            #avg_class_dice = {cls: np.nanmean(dices) for cls, dices in class_dice_per_image.items()}
            #avg_all_dice = np.nanmean(list(avg_class_dice.values()))

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics, **seg_metrics_2,
                "Average_Class_Dice": avg_dice_scores_per_class,
                "average_dice_score": avg_dice_scores,
                
                "AVD_Score_Per_Class": avg_avd_scores_per_class,
                "Average_AVD_Score": avg_avd_scores,
        
                "ACC_Score_Per_Class": avg_acc_scores_per_class,
                "Average_ACC_Score": avg_acc_scores,

                "AVD_Score_Per_Class1": avd_scores_per_class1,
                "Average_AVD_Score1": avg_avd_scores1,

                "IOU_Score_Per_Class": avg_iou_scores_per_class,
                "Average_IOU_Score": avg_iou_scores,

                "ACC_Score_Per_Class1": avd_acc_scores_per_class1,
                "Average_ACC_Score1": avg_acc_scores1,

                # "retfluidnetscore" : dice_per_class
                #"HD_Score_Per_Class": avg_hd_scores_per_class,  # 添加HD到日志中
                #"Average_HD_Score": avg_hd_scores  # 添加平均HD到日志中
            }

        self.write_log_to_txt(log, epoch, "driveunet.txt")    
        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
        self.total_union_2 = 0
        #self.total_pixels=0
        ###self.total_pixels = [0] * self.num_classes  # Initialize as a list with length equal to the number of classes
        #self.total_output=0
        #self.total_target=0
        #self.total_output = torch.zeros_like(output)
        #self.total_target = torch.zeros_like(target)


    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.total_union_2 += union + inter
        ###self.total_pixels = class_pixels
        #self.total_output += output
        #self.total_target += target

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 5),
            "Mean_IoU": np.round(mIoU, 5),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 5)))
        }
    
    def _get_seg_metrics_2(self):
        Dice = 2.0 * self.total_inter / (np.spacing(1) + self.total_union_2)
        mDice = Dice.mean()
        return {
            "Mean_Dice": np.round(mDice, 5),
            "Class_Dice": dict(zip(range(self.num_classes), np.round(Dice, 5)))
        }
    
    

    def write_log_to_txt(self, log, epoch, file_name):
        with open(file_name, "a") as f:
            f.write(f"Epoch {epoch}:\n")
            for k, v in log.items():
                if isinstance(v, list):
                    f.write(f"{k}:\n")
                    for i, val in enumerate(v):
                        f.write(f"  Class {i}: {val}\n")
                else:
                    f.write(f"{k}: {v}\n")
            f.write("\n")



    """
    def _get_seg_metrics_2(self):
        Dice = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if self.total_pixels[i]>0:
                Dice[i] = (2.0 * self.total_inter[i]) / (np.spacing(1) + self.total_inter[i] + self.total_union_2[i])  # Calculate Dice score for each class, excluding class 0
            else:
                Dice = np.NaN

        #if target() >0:
        #    Dice = (2.0 * self.total_inter) / (np.spacing(1) + self.total_inter + self.total_union_2)
        #else:
        #    Dice = np.NaN
        # exclude class 0 from Dice
        #Dice = np.delete(Dice, 0, axis=0)
        mDice = Dice.mean()
        return {
            "Mean_Dice": np.round(mDice, 5),
            "Class_Dice": dict(zip(range(self.num_classes), np.round(Dice, 5)))
        }
    """

    #def __init__(self):
    #    self.total_dice = np.zeros(self.num_classes)
    #    self.Dice_all = np.zeros(self.num_classes)

    #def _get_seg_metrics_2(self, correct, labeled, inter, union,class_pixels):
    """
    def _get_seg_metrics_2(self):
        self.Dice_all = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if i==0:
                self.Dice_all[i] = (2.0 * self.total_inter[i]) / (self.total_union_2[i])
            elif self.total_pixels[i] > 0:
                self.Dice_all[i] = (2.0 * self.total_inter[i]) / (self.total_union_2[i])
            else:
                self.Dice_all[i] = np.NaN    
        
            # accumulate the dice value for each class
            #if not np.isnan(self.Dice_all[i]):
            #    self.total_dice[i] += self.Dice_all[i]
    
        Dice = self.Dice_all[1:] # exclude class 0
        mDice = np.nanmean(Dice)

        return {
            "Mean_Dice": np.round(mDice, 5),
            "Class_Dice": dict(zip(range(self.num_classes-1), np.round(Dice, 5)))
        }
    """


    


