
from torch.utils.data import Dataset



class Dataset(Dataset):
    
    def __init__(self, data, max_length:int, vocab: Callable, transform=None):
        super().__init__()
        
        self.data = data
        self.transform = transform
        self.max_length = max_length
        self.vocab = vocab
        
    def __len__(self):
        raise NotimplementedError("Please implement the __len__ method.")
    
    def __getitem__(self, idx):
        raise NotimplementedError("Please implement the __getitem__ method.")