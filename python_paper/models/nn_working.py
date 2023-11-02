import torch
import numpy as np
import torch.nn as nn

def trainModel(device, training_loader, model, optimizer):
    mseLoss = nn.MSELoss()
    loss_total = 0
    for idx, (input_data, target_data) in enumerate(training_loader):
        batch_size = input_data.size(0)
        input_t = input_data.view(batch_size,-1).to(device)
        encoder_output_t, output_t = model.forward(input_t)

        input_t1 = target_data.to(device)
        encoder_output_t1, output_t1 = model.forward(input_t1)
        
        koopman_t1 = model.koopmanOperation(encoder_output_t)
        loss = mseLoss(input_t, output_t) + mseLoss(input_t1, output_t1) + mseLoss(koopman_t1, encoder_output_t1)
        loss_total = loss_total + loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = 0

    return loss_total