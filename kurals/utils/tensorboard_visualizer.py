"""Class to create Tensorboard Visualization during training"""
from torch.utils.tensorboard import SummaryWriter


class TensorboardMultiLossVisualizer:
    """Class to generate Tensorboard visualisation

    PARAMETERS
    ----------
    writer: SummaryWriter from Tensorboard
    """

    def __init__(self, writer):
        self.writer = writer

    def update_train_loss(self, loss, losses, iteration):
        self.writer.add_scalar('train_losses/global', loss,
                               iteration)
        self.writer.add_scalar('train_losses/CE', losses[0],
                               iteration)
        self.writer.add_scalar('train_losses/Dice', losses[1],
                               iteration)

    def update_multi_train_loss(self, global_loss, rd_loss, rd_losses, iteration):
        self.writer.add_scalar('train_losses/global', global_loss,
                               iteration)
        self.writer.add_scalar('train_losses/range_doppler/global', rd_loss,
                               iteration)
        self.writer.add_scalar('train_losses/range_doppler/CE', rd_losses[0],
                               iteration)
        self.writer.add_scalar('train_losses/range_doppler/Dice', rd_losses[1],
                               iteration)

    def update_val_loss(self, loss, losses, iteration):
        self.writer.add_scalar('val_losses/global', loss,
                               iteration)
        self.writer.add_scalar('val_losses/CE', losses[0],
                               iteration)
        self.writer.add_scalar('val_losses/Dice', losses[1],
                               iteration)

    def update_multi_val_loss(self, global_loss, rd_loss, rd_losses, iteration):
        self.writer.add_scalar('validation_losses/global', global_loss,
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/global', rd_loss,
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/CE', rd_losses[0],
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/Dice', rd_losses[1],
                               iteration)

    def update_learning_rate(self, lr, iteration):
        self.writer.add_scalar('parameters/learning_rate', lr, iteration)

    def update_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('validation_losses/global', metrics['loss'],
                               iteration)
        self.writer.add_scalar('validation_losses/CE', metrics['loss_ce'],
                               iteration)
        self.writer.add_scalar('validation_losses/Dice', metrics['loss_dice'],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Mean', metrics['acc'],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Background',
                               metrics['acc_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Pedestrian',
                               metrics['acc_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Cyclist',
                               metrics['acc_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Car',
                               metrics['acc_by_class'][3],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Mean', metrics['prec'],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Background',
                               metrics['prec_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Pedestrian',
                               metrics['prec_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Cyclist',
                               metrics['prec_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelPrecision/Car',
                               metrics['prec_by_class'][3],
                               iteration)
        self.writer.add_scalar('PixelRecall/Mean', metrics['recall'],
                               iteration)
        self.writer.add_scalar('PixelRecall/Background',
                               metrics['recall_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelRecall/Pedestrian',
                               metrics['recall_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelRecall/Cyclist',
                               metrics['recall_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelRecall/Car',
                               metrics['recall_by_class'][3],
                               iteration)
        self.writer.add_scalar('MIoU/Mean', metrics['miou'],
                               iteration)
        self.writer.add_scalar('MIoU/Background',
                               metrics['miou_by_class'][0],
                               iteration)
        self.writer.add_scalar('MIoU/Pedestrian',
                               metrics['miou_by_class'][1],
                               iteration)
        self.writer.add_scalar('MIoU/Cyclist',
                               metrics['miou_by_class'][2],
                               iteration)
        self.writer.add_scalar('MIoU/Car',
                               metrics['miou_by_class'][3],
                               iteration)

    def update_multi_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('validation_losses/global',
                               metrics['range_doppler']['loss'],
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler/global',
                               metrics['range_doppler']['loss'], iteration)
        self.writer.add_scalar('validation_losses/range_doppler/CE',
                               metrics['range_doppler']['loss_ce'], iteration)
        self.writer.add_scalar('validation_losses/range_doppler/Dice',
                               metrics['range_doppler']['loss_dice'], iteration)

        self.writer.add_scalar('Range_Doppler_metrics/PixelAccuracy',
                               metrics['range_doppler']['acc'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/PixelPrecision',
                               metrics['range_doppler']['prec'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/PixelRecall',
                               metrics['range_doppler']['recall'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/MIoU',
                               metrics['range_doppler']['miou'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/Dice',
                               metrics['range_doppler']['dice'],
                               iteration)

    def update_detection_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('AveragePrecision/Mean', metrics['map'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Pedestrian',
                               metrics['map_by_class']['pedestrian'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Cyclist',
                               metrics['map_by_class']['cyclist'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Car',
                               metrics['map_by_class']['car'],
                               iteration)

    def update_multi_test_metrics(self, metrics, iteration):
        self.writer.add_scalar('test_losses/global',
                               metrics['range_doppler']['loss'],
                               iteration)
        self.writer.add_scalar('test_losses/range_doppler/global',
                               metrics['range_doppler']['loss'], iteration)
        self.writer.add_scalar('test_losses/range_doppler/CE',
                               metrics['range_doppler']['loss_ce'], iteration)
        self.writer.add_scalar('test_losses/range_doppler/Dice',
                               metrics['range_doppler']['loss_dice'], iteration)

        self.writer.add_scalar('Test/Range_Doppler_metrics/PixelAccuracy',
                               metrics['range_doppler']['acc'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/PixelPrecision',
                               metrics['range_doppler']['prec'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/PixelRecall',
                               metrics['range_doppler']['recall'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/MIoU',
                               metrics['range_doppler']['miou'],
                               iteration)
        self.writer.add_scalar('Test/Range_Doppler_metrics/Dice',
                               metrics['range_doppler']['dice'],
                               iteration)

    def update_img_masks(self, pred_grid, gt_grid, iteration):
        self.writer.add_image('Predicted_masks', pred_grid, iteration)
        self.writer.add_image('Ground_truth_masks', gt_grid, iteration)

    def update_multi_img_masks(self, rd_pred_grid, rd_gt_grid, iteration):
        self.writer.add_image('Range_Doppler/Predicted_masks', rd_pred_grid, iteration)
        self.writer.add_image('Range_Doppler/Ground_truth_masks', rd_gt_grid, iteration)
    
    # lt@20240703
    def update_prob_loss(self, dataset, mode, loss_hard, loss_easy, epoch):
        if dataset == 'KuRALS_CW':
            hard_bkg, hard_uav, hard_ped, hard_veh = loss_hard
            easy_bkg, easy_uav, easy_ped, easy_veh = loss_easy
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Bkg_loss', hard_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Bkg_loss', easy_bkg, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_UAV_loss', hard_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_UAV_loss', easy_uav, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Ped_loss', hard_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Ped_loss', easy_ped, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Veh_loss', hard_veh, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Veh_loss', easy_veh, epoch)
        elif dataset == 'KuRALS_PD':
            hard_bkg, hard_uav, hard_ped, hard_car, hard_boa = loss_hard
            easy_bkg, easy_uav, easy_ped, easy_car, easy_boa = loss_easy
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Bkg_loss', hard_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Bkg_loss', easy_bkg, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_UAV_loss', hard_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_UAV_loss', easy_uav, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Ped_loss', hard_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Ped_loss', easy_ped, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Car_loss', hard_car, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Car_loss', easy_car, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Boa_loss', hard_boa, epoch)
            self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Boa_loss', easy_boa, epoch)
            
    
    # lt@20240704
    def update_prob_loss_iteration(self, mode, loss_hard, loss_easy, epoch):
        hard_bkg, hard_uav, hard_ped, hard_veh = loss_hard
        easy_bkg, easy_uav, easy_ped, easy_veh = loss_easy
        
        self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Bkg_loss_iter', hard_bkg, epoch)
        self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Bkg_loss_iter', easy_bkg, epoch)
        
        self.writer.add_scalar(f'{mode}/Prob_loss/Hard_UAV_loss_iter', hard_uav, epoch)
        self.writer.add_scalar(f'{mode}/Prob_loss/Easy_UAV_loss_iter', easy_uav, epoch)
        
        self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Ped_loss_iter', hard_ped, epoch)
        self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Ped_loss_iter', easy_ped, epoch)
        
        self.writer.add_scalar(f'{mode}/Prob_loss/Hard_Veh_loss_iter', hard_veh, epoch)
        self.writer.add_scalar(f'{mode}/Prob_loss/Easy_Veh_loss_iter', easy_veh, epoch)
    
    # lt@20240703
    def update_prob_grad(self, dataset, mode, grad_pos_hard, grad_pos_easy, grad_neg_hard, 
                                     grad_neg_easy, epoch):
        if dataset == 'KuRALS_CW':
            pos_hard_bkg, pos_hard_uav, pos_hard_ped, pos_hard_veh = grad_pos_hard
            pos_easy_bkg, pos_easy_uav, pos_easy_ped, pos_easy_veh = grad_pos_easy
            neg_hard_bkg, neg_hard_uav, neg_hard_ped, neg_hard_veh = grad_neg_hard
            neg_easy_bkg, neg_easy_uav, neg_easy_ped, neg_easy_veh = grad_neg_easy
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_Bkg_grad', pos_hard_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_Bkg_grad', pos_easy_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_Bkg_grad', neg_hard_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_Bkg_grad', neg_easy_bkg, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_UAV_grad', pos_hard_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_UAV_grad', pos_easy_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_UAV_grad', neg_hard_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_UAV_grad', neg_easy_uav, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_Ped_grad', pos_hard_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_Ped_grad', pos_easy_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_Ped_grad', neg_hard_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_Ped_grad', neg_easy_ped, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_Veh_grad', pos_hard_veh, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_Veh_grad', pos_easy_veh, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_Veh_grad', neg_hard_veh, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_Veh_grad', neg_easy_veh, epoch)
        elif dataset == 'KuRALS_PD':
            pos_hard_bkg, pos_hard_uav, pos_hard_ped, pos_hard_car, pos_hard_boa = grad_pos_hard
            pos_easy_bkg, pos_easy_uav, pos_easy_ped, pos_easy_car, pos_easy_boa = grad_pos_easy
            neg_hard_bkg, neg_hard_uav, neg_hard_ped, neg_hard_car, neg_hard_boa = grad_neg_hard
            neg_easy_bkg, neg_easy_uav, neg_easy_ped, neg_easy_car, neg_easy_boa = grad_neg_easy
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_Bkg_grad', pos_hard_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_Bkg_grad', pos_easy_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_Bkg_grad', neg_hard_bkg, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_Bkg_grad', neg_easy_bkg, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_UAV_grad', pos_hard_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_UAV_grad', pos_easy_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_UAV_grad', neg_hard_uav, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_UAV_grad', neg_easy_uav, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_Ped_grad', pos_hard_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_Ped_grad', pos_easy_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_Ped_grad', neg_hard_ped, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_Ped_grad', neg_easy_ped, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_Car_grad', pos_hard_car, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_Car_grad', pos_easy_car, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_Car_grad', neg_hard_car, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_Car_grad', neg_easy_car, epoch)
            
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Hard_Boa_grad', pos_hard_boa, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Pos_Easy_Boa_grad', pos_easy_boa, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Hard_Boa_grad', neg_hard_boa, epoch)
            self.writer.add_scalar(f'{mode}/Prob_grad/Neg_Easy_Boa_grad', neg_easy_boa, epoch)