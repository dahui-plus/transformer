from torch.utils.data import Dataset
from datasets import load_dataset
import os


class Data(Dataset):
  def __init__(self, path: str) -> None:
    super(Dataset, self).__init__()
    self.path = path
    assert os.path.isfile(self.path)
    
    with open(self.path, encoding='utf-8') as file:
      text_pair_list = file.readlines()
      
    self.target_list = [text.strip().removesuffix('\n').split('\t')[0] for text in text_pair_list]
    self.source_list = [text.strip().removesuffix('\n').split('\t')[1] for text in text_pair_list]
    
  def __getitem__(self, index: int) -> list[str, str]:
    return self.source_list[index], self.target_list[index]
  
  def __len__(self):
    return len(self.source_list)


class QADataset(Dataset):
  def __init__(self, path: str, cache_dir: str) -> None:
    super(QADataset, self).__init__()
    dataset = load_dataset(path=path, cache_dir=cache_dir)
    self.dataset = dataset.data['train'].to_pylist()

  def __getitem__(self, index) -> list[str, str]:
    return self.dataset[index]['question'], self.dataset[index]['response']

  def __len__(self) -> int:
    return len(self.dataset)


if __name__ == '__main__':
  data = Data(path='./dataset/spa-eng/spa.txt')
  for s, t in data[:2]:
    print(s, t)
