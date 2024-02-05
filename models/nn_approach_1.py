import torch
import numpy as np
import torch.nn as nn
from plots.plots import plotPredictions
from sklearn.metrics import mean_squared_error

def trainAE(epoch, device, training_loader, s, model, a1, a2, a3, a4, optimizer):
  mseLoss = nn.MSELoss()
  loss_total = 0

  for idx, (input_data, target_data) in enumerate(training_loader):
        batch_size = input_data.size(0)
        input_t = input_data.view(batch_size,-1).to(device)
        encoder_output_t, output_t = model.forward(input_t)

        loss_rec = mseLoss(input_t, output_t)
        model.kMatrixDiag.requires_grad = False
        model.kMatrixUT.requires_grad = False
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
        input_t = input_data.view(batch_size,-1).to(device) # size [16,2]
        encoder_output_t, output_t = model.forward(input_t) # size of encoder [16,3] output [16,2]

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
        
        model.kMatrixDiag.requires_grad = True
        model.kMatrixUT.requires_grad = True
        loss = a1*loss_rec + a2*loss_lin + a3*loss_pred + a4*l2_norm

        loss = a1*loss_rec + a2*loss_lin + a3*loss_pred + a4*l2_norm
        print(idx, "debug loss", loss)
        loss_total = loss_total + loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = 0
    return loss_rec, loss_total, model.getKoopmanMatrix()

def testModel(model, device, testing_loader):
    model.eval()
    test_loss = 0
    mseLoss = nn.MSELoss()
    predicted_dataset = np.zeros((7472,6))
    target_dataset = np.zeros((7472,6))

    for mbidx, (input_data, target_data) in enumerate(testing_loader):
        batch_size = input_data.size(0)
        input_t = input_data.view(batch_size, -1).to(device) # size [16,5]
        encoder_output_t, output_t = model.forward(input_t) 
        input_data_tnext = target_data.to(device)            # size [16,5]

        koopman_tnext = model.koopmanOperation(encoder_output_t,1)
        yPred = model.recover(koopman_tnext)
        test_loss = test_loss + mseLoss(input_data_tnext, yPred)

        # rmspe = mseLoss(input_data_tnext/input_data_tnext, yPred/input_data_tnext)
        # print(f"%loss : {rmspe}")

        predicted_dataset[mbidx:mbidx+yPred.size(0)] = yPred
        target_dataset[mbidx:mbidx+input_data_tnext.size(0)] = input_data_tnext

    tar = target_dataset[:,0]    
    pred = predicted_dataset[:,0]

    C = tar[0]
    soh_tar = tar/C
    soh_pred = pred/C
    rms = np.sqrt(mean_squared_error(soh_tar,soh_pred))

    print(f"final rmse error of SOH {rms}")
    
    return test_loss, predicted_dataset, target_dataset, soh_pred, soh_tar