import torch
from transformers import BertTokenizer
import os

from model import Transformer


d_model = 128
num_layers = 4
num_heads = 4
dff = 256
dropout_rate = 0.1
max_sequence_length = 100
weight_file = './checkpoints/model-0.pt'
cache_dir = './dataset'
data_path = './dataset/Open-Orca___open_orca/default'
tokens_path = './dataset/ted_hrlr_translate_pt_en_converter/assets/en_vocab.txt'


def qa(source: str):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  tokenizer = BertTokenizer(vocab_file=tokens_path)
  model = Transformer(vocab_size=tokenizer.vocab_size+2, d_model=d_model,
                      num_layers=num_layers, num_heads=num_heads, dff=dff,
                      target_vocab_size=tokenizer.vocab_size+2,
                      padding_idx=tokenizer.pad_token_id, dropout_rate=dropout_rate)
  model.to(device=device)
  
  if os.path.isfile(weight_file):
    model.load_state_dict(torch.load(weight_file))
    
  source_tokens = torch.tensor(tokenizer.encode(source, max_length=1024, truncation=True), device=device).unsqueeze(0)
  source_padding_mask = torch.ones(size=source_tokens.shape, device=device) == tokenizer.pad_token_id
  
  start_token = torch.tensor([tokenizer.cls_token_id], device=device).unsqueeze(0)
  start_padding_mask = torch.ones(start_token.shape, device=device) == tokenizer.pad_token_id
  target_tokens = start_token
  
  model.eval()
  for _ in range(max_sequence_length):
    predict = model(source_tokens, target_tokens, source_padding_mask, start_padding_mask)
    
    start_token = torch.argmax(predict, dim=-1)
    token = start_token.squeeze(0)[-1]
    target_tokens = torch.concat([target_tokens.view(-1), token.view(-1)], dim=0).unsqueeze(0)
    start_padding_mask = torch.ones(size=target_tokens.shape, device=device)
    
    if token == tokenizer.sep_token_id:
      break

  return tokenizer.decode(target_tokens.squeeze(0))


if __name__ == '__main__':
  answer = qa("Write a question about the following article: Coming off their home win over the Buccaneers, the Packers flew to Ford Field for a Week 12 Thanksgiving duel with their NFC North foe, the Detroit Lions. After a scoreless first quarter, Green Bay delivered the game's opening punch in the second quarter with quarterback Aaron Rodgers finding wide receiver Greg Jennings on a 3-yard touchdown pass. The Packers added to their lead in the third quarter with a 1-yard touchdown run from fullback John Kuhn, followed by Rodgers connecting with wide receiver James Jones on a 65-yard touchdown pass and a 35-yard field goal from kicker Mason Crosby. The Lions answered in the fourth quarter with a 16-yard touchdown run by running back Keiland Williams and a two-point conversion pass from quarterback Matthew Stafford to wide receiver Titus Young), yet Green Bay pulled away with Crosby nailing a 32-yard field goal. Detroit closed out the game with Stafford completing a 3-yard touchdown pass to wide receiver Calvin Johnson. With the win, the Packers acquired their first 11-0 start in franchise history, beating the 1962 team which started 10-0 and finished 14-1 including postseason play. Rodgers (22/32 for 307 yards, 2 TDs) was named NFL on FOX's 2011 Galloping Gobbler Award Winner. Question about the article:")
  print(answer)
