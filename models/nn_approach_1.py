import torch
import random
import numpy as np
import torch.nn as nn
from ..plot_functions.plots import plot_eqn, plot_soc, plot_stock

def inverse_scaler(test_scaler, actual_list, predicted_list):
    # print('HIIIIIIIIIIIIIIIIIIIIIIIII')
    # print(predicted_list.shape)
    stock_predicted = np.zeros((predicted_list.shape[0], predicted_list.shape[2]))
    for i in range(predicted_list.shape[0]):
        stock_predicted[i] = predicted_list[i,-1,:]
    
    num_samples, num_features = actual_list.shape[0], actual_list.shape[1]
    actual_flattened = actual_list.reshape(num_samples, num_features)
    predicted_flattened = stock_predicted.reshape(num_samples, num_features)
    actual_iw_scale = test_scaler.inverse_transform(actual_flattened)
    predicted_iw_scale = test_scaler.inverse_transform(predicted_flattened)
    actual_og_scale = actual_iw_scale.reshape(num_samples, num_features)
    predicted_og_scale = predicted_iw_scale.reshape(num_samples, num_features)
    
    return actual_og_scale[:,-1], predicted_og_scale[:,-1]

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
            loss_rec = loss_rec+mseLoss(input_tnext, output_tnext)
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
        print(idx, "debug loss", loss)
        loss_total = loss_total + loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = 0

    return loss_rec, loss_total, model.getKoopmanMatrix()

def testModel(exp, test_key, epoch, device, model, testing_loader, plot_dir, test_scaler):
    model.eval()
    test_loss = 0
    mseLoss = nn.MSELoss()

    actual_list = []
    predicted_list = []
    mm_scale_actual_list = []
    mm_scale_pred_list = []

    # random_idx = random.randint(0, len(testing_loader)-1)
    
    for idx, (input_data, target_data) in enumerate(testing_loader):
        batch_size = target_data.size(0)
        sequence_size = target_data.size(1)
        feature_size = target_data.size(2)

        actual_values = target_data.to(device)
        predicted_values = torch.zeros((batch_size, sequence_size, feature_size)).to(device)
        
        input_t = input_data.view(batch_size, -1).to(device)
        encoder_output_t, output_t = model(input_t)
        input_data_tnext = target_data.to(device) 
        test_loss = 0
        
        for s in range(input_data_tnext.shape[1]):
            input_tnext = input_data_tnext[:, s, :].view(batch_size, -1).to(device) # [16,2]
            koopman_tnext = model.koopmanOperation(encoder_output_t, s+1)
            recover_tnext = model.recover(koopman_tnext)
            predicted_values[:,s,:] = recover_tnext
            test_loss = test_loss + mseLoss(input_tnext, recover_tnext)

        actual_list.append(input_data[:,:].cpu().numpy())
        predicted_list.append(predicted_values[:,:,:].cpu().numpy())
        
        mm_scale_actual_list.append(input_data[:,-1].cpu().numpy())
        mm_scale_pred_list.append(predicted_values[:,:,-1].cpu().numpy())
        
        # if idx==random_idx:
        #     if exp=='eqn':
        #         plot_eqn(device, test_key, epoch, idx, actual_values, predicted_values, plot_dir)
        #     elif exp=='soc':
        #         plot_soc(device, test_key, epoch, idx, actual_values, predicted_values, plot_dir)
        #     else:
        #         plot_stock(device, test_key, epoch, idx, actual_values, predicted_values, plot_dir)

    mm_scale_actual = np.concatenate(mm_scale_actual_list)
    mm_scale_predicted = np.concatenate(mm_scale_pred_list)
    actual_values = np.concatenate(actual_list)
    predicted_values = np.concatenate(predicted_list, axis = 0)

    iw_scale_actual, iw_scale_predicted = inverse_scaler(test_scaler, actual_values, predicted_values)
    return test_loss, mm_scale_actual, mm_scale_predicted, iw_scale_actual, iw_scale_predicted