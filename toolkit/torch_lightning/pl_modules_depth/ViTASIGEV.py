import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import numpy as np
# from torch import nn
import time

from toolkit.function.base_function import save_disp_results,InputPadder,disp_D1_vis
from toolkit.function.evaluator import calcu_EPE,calcu_PEP,calcu_D1all
from toolkit.torch_lightning.lightning_function import schedule_select
from toolkit.function.models import prepare_model

from toolkit.function.depth_function.loss_fuction import Main_Loss # ,Geometry_Loss

class ViTASIGEV(LightningModule):
    def __init__(self, hparams):
        super(ViTASIGEV, self).__init__()   
        self.save_hyperparameters()
        # model
        self.model = prepare_model(hparams)        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.validation_metric = {'step_EPE':[],'step_PEP1':[],'step_PEP2':[],'step_PEP3':[],'step_D1all':[]}
        self.test_metric = {'step_EPE':[],'step_PEP1':[],'step_PEP2':[],'step_PEP3':[],'step_D1all':[],'step_time':[]}
        self.loss_function = Main_Loss(TopK = hparams.TopK)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.hparams['hparams'].lr,weight_decay=1e-5)
        lr_scheduler = schedule_select(optimizer,self.hparams['hparams'])
        return [optimizer],[lr_scheduler]

    def training_step(self, batch, batch_idx):

        imgL   = batch['left']
        imgR   = batch['right']
        disp_true = batch['disp']
        depth_L = batch['append']['left_depth'].unsqueeze(1) # [B,C,H,W]
        depth_R = batch['append']['right_depth'].unsqueeze(1)
                
        padder = InputPadder(imgL.shape,64, mode = 'replicate')
        [imgL, imgR],_,_ = padder.pad(imgL, imgR)
        (disp_init, disp_preds) = self.model(imgL, imgR, iters=12)
        (r_disp_init_ast, r_disps_ast) = self.model(imgR.flip(-1), imgL.flip(-1), iters=12) 
        imgL = padder.unpad(imgL)
        imgR = padder.unpad(imgR)
        disp_init = [padder.unpad(disp_init)]
        r_disp_init_ast = [padder.unpad(r_disp_init_ast.flip(-1)).flip(-1)]
    
        disp_preds = [padder.unpad(i) for i in disp_preds]
        r_disps_ast = [padder.unpad(i.flip(-1)).flip(-1) for i in r_disps_ast]
        disps = disp_preds[-5:-1]
        r_disps_ast = r_disps_ast[-5:-1]

        loss = self.loss_function.forward(disp_init,r_disp_init_ast,imgL, imgR, depth_L, depth_R, self.hparams['hparams'])
        for i in range(len(disps)):
            loss +=  self.loss_function.forward([disps[i]],[r_disps_ast[i]],imgL, imgR, depth_L, depth_R, self.hparams['hparams']) # input in [B,C,H，W]
        
        self.log("loss_step", round(loss.item(),3), prog_bar=True, on_step=True)
        if torch.isnan(loss).any():
            print('nan exists in:', batch['left_dir'])
        self.training_step_outputs.append(loss.item())

        disp_true = batch['disp'].squeeze()
        mask = (disp_true > 0)&(disp_true<self.hparams['hparams'].max_disp)
        mask.detach_()
        
        if not torch.sum(disp_true).item() == 0:
            epe = calcu_EPE(disps[-1].squeeze()[mask], disp_true[mask])
            self.log("EPE_step", round(epe.item(),3), prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self):
        self.training_step_outputs = np.array(self.training_step_outputs)[~np.isnan(self.training_step_outputs)]
        train_epoch_loss = np.mean(self.training_step_outputs)
        self.log('loss_epoch', train_epoch_loss) 
        self.training_step_outputs = []# free memory

    def validation_step(self, batch, batch_idx):
        
        imgL   = batch['left']
        imgR   = batch['right']
        disp_true = batch['disp']

        padder = InputPadder(imgL.shape,64, mode = 'replicate')
        [imgL, imgR],_,_ = padder.pad(imgL, imgR)
        
        pred = self.model(imgL, imgR, iters=32, test_mode=True)
        pred = padder.unpad(pred.squeeze(1))

        mask = (disp_true > 0)&(disp_true<self.hparams['hparams'].max_disp)
        mask.detach_()
        self.validation_metric['step_EPE'].append(calcu_EPE(pred[mask], disp_true[mask]).item())
        self.validation_metric['step_D1all'].append(calcu_D1all(pred[mask], disp_true[mask]).item())
        
    def on_validation_epoch_end(self):
        self.validation_metric['step_EPE'] = np.array(self.validation_metric['step_EPE'])[~np.isnan(self.validation_metric['step_EPE'])]
        valid_epoch_EPE = np.mean(self.validation_metric['step_EPE'])
        self.log('valid_epoch_EPE', valid_epoch_EPE, sync_dist=True) 
        self.validation_metric['step_EPE'] = []# free memory
        
        self.validation_metric['step_D1all'] = np.array(self.validation_metric['step_D1all'])[~np.isnan(self.validation_metric['step_D1all'])]
        valid_epoch_D1 = np.mean(self.validation_metric['step_D1all'])
        self.log('valid_epoch_D1', valid_epoch_D1, sync_dist=True) 
        self.validation_metric['step_D1all'] = []# free memory

    def test_step(self, batch, batch_idx): 
        
        imgL   = batch['left']
        imgR   = batch['right']
        disp_true = batch['disp'] # [B,H,W]

        padder = InputPadder(imgL.shape,64, mode = 'replicate')
        [imgL, imgR],_,_ = padder.pad(imgL, imgR)
        
        pred = self.model(imgL, imgR, iters=32, test_mode=True)
        pred = padder.unpad(pred.squeeze(1))
        
        if torch.mean(disp_true)==0:
            disp_true = pred
        mask = (disp_true > 0)&(disp_true<self.hparams['hparams'].max_disp) 
        self.test_metric['step_EPE'].append(calcu_EPE(pred[mask], disp_true[mask]).item())
        self.test_metric['step_PEP1'].append(calcu_PEP(pred[mask], disp_true[mask],thr=1).item())
        self.test_metric['step_PEP2'].append(calcu_PEP(pred[mask], disp_true[mask],thr=2).item())
        self.test_metric['step_PEP3'].append(calcu_PEP(pred[mask], disp_true[mask],thr=3).item())
        self.test_metric['step_D1all'].append(calcu_D1all(pred[mask], disp_true[mask]).item())
        
        pred_disp = pred.squeeze().detach().cpu().numpy()
        
        # visualize disp
        save_disp_results(batch['save_dir_disp'][0],batch['save_dir_disp_vis'][0],pred_disp,display=True)

    def on_test_epoch_end(self):
        results = {}
        results['EPE'] = np.mean(self.test_metric['step_EPE'])
        results['PEP1']  = np.mean(self.test_metric['step_PEP1'])
        results['PEP2']  = np.mean(self.test_metric['step_PEP2'])
        results['PEP3']  = np.mean(self.test_metric['step_PEP3'])
        results['D1all']  = np.mean(self.test_metric['step_D1all'])
        print(results)