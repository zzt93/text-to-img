
import minbpe.regex
import torch
from torch.nn.utils.rnn import pad_sequence

tokenizer = minbpe.regex.RegexTokenizer()
a=tokenizer.encode('数字5(19022) [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]')


sequences = [torch.tensor([230, 149, 176, 229, 173, 151, 53, 40, 49, 57, 48, 50, 50, 41, 32, 91, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 44, 32, 45, 48, 46, 53, 93]),torch.tensor([230, 149, 176, 229, 173, 151, 49, 40, 56, 56, 57, 49, 41, 32, 91, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 44, 32, 48, 46, 50, 53, 44, 32, 45, 48, 46, 50, 53, 93])]
sequences = pad_sequence(sequences, batch_first=True, padding_value=-1)
mask = (sequences != -1).bool()