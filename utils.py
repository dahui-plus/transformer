from torch import Tensor, sum, argmax, float32, save
import numpy as np
import os



def masked_loss(predict: Tensor, label: Tensor, loss_fn):
  # mask = label != 0
  loss = loss_fn(predict.permute(0, 2, 1), label)
  
  # mask = mask.to(dtype=loss.dtype)
  # loss *= mask
  
  # loss = sum(loss) / sum(mask)
  return loss


def masked_accuracy(predict: Tensor, label: Tensor):
  predict = argmax(predict, dim=-1)
  label = label.to(dtype=predict.dtype)
  
  match = label == predict
  mask = label != 0
  match = match & mask
  
  match = match.to(dtype=float32)
  mask = mask.to(dtype=float32)
  
  return sum(match) / sum(mask)


class EarlyStopping:
  def __init__(self, save_path, patience=10, verbose=False, delta=0):
    self.save_path = save_path
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta

  def __call__(self, val_loss, state_dict):
    score = -val_loss

    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(val_loss, state_dict)
    elif score < self.best_score + self.delta:
      self.counter += 1
      print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
          self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, state_dict)
      self.counter = 0

  def save_checkpoint(self, val_loss, state_dict):
    '''Saves model when validation loss decrease.'''
    if self.verbose:
      print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    path = os.path.join(self.save_path, 'model.pt')
    save(state_dict, path)	# 这里会存储迄今最优模型的参数
    self.val_loss_min = val_loss
