import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime

from model import Transformer
from dataset import QADataset
from utils import masked_loss, masked_accuracy, EarlyStopping


seed = 42
split_ratio = .9
batch_size = 8
d_model = 128
num_layers = 4
num_heads = 4
dff = 256
dropout_rate = 0.1
num_epoch = 100
learning_rate = 0.0001
save_interval = 30 * 60 # half an hour
weight_file = './checkpoints/model-qa.pt'
log_dir = './logs'
weight_path = './checkpoints'
cache_dir = './dataset'
data_path = './dataset/Open-Orca___open_orca/default'
tokens_path = './dataset/ted_hrlr_translate_pt_en_converter/assets/en_vocab.txt'

if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  generator = torch.Generator().manual_seed(seed)
  
  dataset = QADataset(path=data_path, cache_dir=cache_dir)
  tokenizer = BertTokenizer(vocab_file=tokens_path)

  train_length = int(len(dataset) * split_ratio)
  test_length = len(dataset) - train_length
  
  train_dataset, test_dataset = random_split(dataset, lengths=[train_length, test_length], generator=generator)
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
  test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
  
  model = Transformer(vocab_size=tokenizer.vocab_size+2, d_model=d_model,
                      num_layers=num_layers, num_heads=num_heads, dff=dff,
                      target_vocab_size=tokenizer.vocab_size+2,
                      padding_idx=tokenizer.pad_token_id, dropout_rate=dropout_rate)
  model.to(device=device)
  if os.path.isfile(weight_file):
    model.load_state_dict(torch.load(weight_file))
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  early_stop = EarlyStopping(save_path=weight_path)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=num_epoch*len(train_dataset), eta_min=1e-8,verbose=True)
  writer = SummaryWriter(log_dir)
  start_date = datetime.now()

  for epoch in range(num_epoch):
    train_loop = tqdm(train_dataloader, ncols=100)
    model.train()
    total_loss = total_accuracy = 0
    
    for (source, target) in train_loop:
      source_pack = tokenizer.batch_encode_plus(source, padding=True, max_length=1024, truncation=True)
      target_pack = tokenizer.batch_encode_plus(source, padding=True, max_length=1024, truncation=True)
      
      source_tokens, source_padding_mask = source_pack['input_ids'], source_pack['attention_mask']
      target_tokens, target_padding_mask = target_pack['input_ids'], target_pack['attention_mask']
      
      source_tokens = torch.tensor(source_tokens, device=device)
      target_tokens = torch.tensor(target_tokens, device=device)
      source_padding_mask = torch.tensor(source_padding_mask, device=device) == tokenizer.pad_token_id
      target_padding_mask = torch.tensor(target_padding_mask, device=device) == tokenizer.pad_token_id
 
      predict = model(source_tokens, target_tokens[:, :-1], source_padding_mask, target_padding_mask[:, :-1])
      
      optimizer.zero_grad()
      loss = masked_loss(predict, target_tokens[:, 1:], loss_fn)
      accuracy = masked_accuracy(predict, target_tokens[:, 1:])
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      total_accuracy += accuracy.item()
      
      train_loop.set_description(f'[TRAIN-EPOCH {epoch+1}/{num_epoch}]')
      train_loop.set_postfix({'loss': loss.item(), 'accuracy': accuracy.item()})
      
      if (datetime.now() - start_date).seconds >= save_interval:
        torch.save(model.state_dict(), f'demo01/checkpoints/model-{epoch}.pt')
        start_date = datetime.now()
    
    scheduler.step()
    train_loop.write('Train Loss: {}, Accuracy: {}'.format(total_loss/len(train_dataloader),
                                                           total_accuracy/len(train_dataset)))
    writer.add_scalar(tag='train/loss', scalar_value=total_loss/len(train_dataloader), global_step=epoch+1)
    writer.add_scalar(tag='train/accuracy', scalar_value=total_accuracy/len(train_dataloader), global_step=epoch+1)
    
    
    test_loop = tqdm(test_dataloader, ncols=100)
    model.eval()
    total_loss = total_accuracy = 0
    
    for (source, target) in test_loop:
      source_pack = tokenizer.batch_encode_plus(source, padding=True)
      target_pack = tokenizer.batch_encode_plus(target, padding=True)
      
      source_tokens, source_padding_mask = source_pack['input_ids'], source_pack['attention_mask']
      target_tokens, target_padding_mask = target_pack['input_ids'], target_pack['attention_mask']
      
      source_tokens = torch.tensor(source_tokens, device=device)
      target_tokens = torch.tensor(target_tokens, device=device)
      source_padding_mask = torch.tensor(source_padding_mask, device=device) == tokenizer.pad_token_id
      target_padding_mask = torch.tensor(target_padding_mask, device=device) == tokenizer.pad_token_id
      
      with torch.no_grad():
        predict = model(source_tokens, target_tokens[:, :-1], source_padding_mask, target_padding_mask[:, :-1])
        
        loss = masked_loss(predict, target_tokens[:, 1:], loss_fn)
        accuracy = masked_accuracy(predict, target_tokens[:, 1:])
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        
      test_loop.set_description(f'[TEST-EPOCH {epoch+1}/{num_epoch}]')
      test_loop.set_postfix({'loss': loss.item(), 'accuracy': accuracy.item()})
      
    early_stop(val_loss=total_loss/len(test_dataloader), state_dict=model.state_dict())
    test_loop.write('Test Loss: {}, Accuracy: {}'.format(total_loss/len(test_dataloader),
                                                          total_accuracy/len(test_dataset)))
    writer.add_scalar(tag='test/loss', scalar_value=total_loss/len(test_dataloader), global_step=epoch+1)
    writer.add_scalar(tag='test/accuracy', scalar_value=total_accuracy/len(test_dataloader), global_step=epoch+1)
    
    if early_stop.early_stop:
      break
