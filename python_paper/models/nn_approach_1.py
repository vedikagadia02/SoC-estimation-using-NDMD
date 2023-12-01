import torch
import numpy as np
import torch.nn as nn

def trainAE(epoch, device, training_loader, s, model, a1, a2, a3, a4, optimizer):
  mseLoss = nn.MSELoss()
  loss_total = 0

  for idx, (input_data, target_data) in enumerate(training_loader):
        batch_size = input_data.size(0)
        input_t = input_data.view(batch_size,-1).to(device)
        encoder_output_t, output_t = model.forward(input_t)

        loss_rec = mseLoss(input_t, output_t)
        loss_rec.detach()
        loss_rec.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_total = loss_total + loss_rec

  return loss_total

def trainModel(epoch, device, training_loader, s, model, a1, a2, a3, a4, optimizer):
    mseLoss = nn.MSELoss()
    loss_total = 0
    for idx, (input_data, target_data) in enumerate(training_loader):
        batch_size = input_data.size(0)
        input_t = input_data.view(batch_size,-1).to(device)
        encoder_output_t, output_t = model.forward(input_t)

        input_data_tnext = target_data.to(device)
        # encoder_output_t1, output_t1 = model.forward(input_t1)
        loss_rec = mseLoss(input_t, output_t)
        loss_lin = 0
        loss_pred = 0
        for s in range(input_data_tnext.shape[1]):
            input_tnext = input_data_tnext[:,s,:].view(batch_size, -1).to(device) 
            encoder_output_tnext, output_tnext = model.forward(input_tnext) 
            koopman_tnext = model.koopmanOperation(encoder_output_t, s+1)
            recover_tnext = model.recover(koopman_tnext)
            loss_rec = loss_rec+mseLoss(input_tnext,output_tnext)
            loss_lin = loss_lin + mseLoss(koopman_tnext, encoder_output_tnext)
            loss_pred = loss_pred + mseLoss(input_tnext, recover_tnext)
        loss_rec = loss_rec/s
        loss_lin = loss_lin/s
        loss_pred = loss_pred/s

        l2_norm = sum(p.abs().pow(2.0).sum() for p in model.parameters())
        # loss = mseLoss(input_t, output_t) + mseLoss(input_t1, output_t1) + mseLoss(koopman_t1, encoder_output_t1)
        
        if epoch <= 5:
            model.kMatrixDiag.requires_grad = False
            model.kMatrixUT.requires_grad = False
            loss = loss_rec
        else:
            model.kMatrixDiag.requires_grad = True
            model.kMatrixUT.requires_grad = True
            loss = a1*loss_rec + a2*loss_lin + a3*loss_pred + a4*l2_norm

        loss = a1*loss_rec + a2*loss_lin + a3*loss_pred + a4*l2_norm
        loss_total = loss_total + loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = 0
    return loss_rec, loss_total, model.getKoopmanMatrix()