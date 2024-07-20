import torch
from transformers import BertTokenizer
import os

from model import Transformer


seed = 42
d_model = 128
num_layers = 4
num_heads = 8
dff = 512
dropout_rate = 0.1
max_sequence_length = 100
weight_path = './checkpoints/model.pt'
source_tokens_path = './dataset/ted_hrlr_translate_pt_en_converter/assets/pt_vocab.txt'
target_tokens_path = './dataset/ted_hrlr_translate_pt_en_converter/assets/en_vocab.txt'


def translate(source: str):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  source_tokenizer = BertTokenizer(vocab_file=source_tokens_path)
  target_tokenizer = BertTokenizer(vocab_file=target_tokens_path)
  
  model = Transformer(vocab_size=source_tokenizer.vocab_size+2, d_model=d_model,
                      num_layers=num_layers, num_heads=num_heads, dff=dff,
                      target_vocab_size=target_tokenizer.vocab_size+2,
                      padding_idx=source_tokenizer.pad_token_id, dropout_rate=dropout_rate)
  model.to(device=device)
  
  if os.path.isfile(weight_path):
    model.load_state_dict(torch.load(weight_path))

  start_token = torch.tensor([target_tokenizer.cls_token_id], device=device).unsqueeze(0)
  start_padding_mask = torch.ones(start_token.shape, device=device) == target_tokenizer.pad_token_id
  
  source_tokens = torch.tensor(source_tokenizer.encode(source), device=device).unsqueeze(0)
  source_padding_mask = torch.ones(size=source_tokens.shape, device=device) == source_tokenizer.pad_token_id
  target_tokens = start_token

  # result = []
  model.eval()
  for _ in range(max_sequence_length):
    predict = model(source_tokens, target_tokens, source_padding_mask, start_padding_mask)
    
    start_token = torch.argmax(predict, dim=-1)
    token = start_token.squeeze(0)[-1]
    target_tokens = torch.concat([target_tokens.view(-1), token.view(-1)], dim=0).unsqueeze(0)
    start_padding_mask = torch.ones(size=target_tokens.shape, device=device)
    # result.append(target_tokenizer.decode(token))
    
    if token == target_tokenizer.sep_token_id:
      break

  return target_tokenizer.decode(target_tokens.squeeze(0))


if __name__ == '__main__':
  # so i'll just share with you some stories very quickly of some magical things that have happened.
  result = translate('vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.')
  print(f'<<<<<{result}>>>>\n')

  # this is a problem we have to solve .
  result = translate('este é um problema que temos que resolver.')
  print(f'<<<<<{result}>>>>\n')

  # and my neighboring homes heard about this idea .
  result = translate('os meus vizinhos ouviram sobre esta ideia.')
  print(f'<<<<<{result}>>>>\n')
  
  # this is the first book i've ever done.
  result = translate('este é o primeiro livro que eu fiz.')
  print(f'<<<<<{result}>>>>\n')
  
  # hello.
  result = translate('olá.')
  print(f'<<<<<{result}>>>>\n')
