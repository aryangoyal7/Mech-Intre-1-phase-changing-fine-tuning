import torch
from torch.utils.data import Dataset, DataLoader

class ModularArithmeticDataset(Dataset):
    def __init__(self, p, operator, seed=42):
        """
        operator: '+' or '-'
        Format of sequence: [a, op_token, b, eq_token] -> target: c
        Vocab: 0 to P-1 for numbers.
        P for '+'
        P+1 for '-'
        P+2 for '='
        """
        self.p = p
        self.operator = operator
        torch.manual_seed(seed)
        
        self.data = []
        self.targets = []
        
        op_token = p if operator == '+' else p + 1
        eq_token = p + 2
        
        for a in range(p):
            for b in range(p):
                seq = [a, op_token, b, eq_token]
                if operator == '+':
                    c = (a + b) % p
                elif operator == '-':
                    c = (a - b) % p
                else:
                    raise ValueError(f"Unknown operator {operator}")
                
                self.data.append(seq)
                self.targets.append(c)
                
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_dataloaders(p=113, batch_size=256, train_split=0.5):
    # Returns train and test dataloaders for both tasks
    dataset_add = ModularArithmeticDataset(p, '+')
    dataset_sub = ModularArithmeticDataset(p, '-')
    
    # Split into train/test
    def split(dataset):
        train_size = int(len(dataset) * train_split)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        )
        
    train_dl_add, test_dl_add = split(dataset_add)
    train_dl_sub, test_dl_sub = split(dataset_sub)
    
    return {
        '+': {'train': train_dl_add, 'test': test_dl_add, 'full': DataLoader(dataset_add, batch_size=batch_size, shuffle=False)},
        '-': {'train': train_dl_sub, 'test': test_dl_sub, 'full': DataLoader(dataset_sub, batch_size=batch_size, shuffle=False)}
    }
