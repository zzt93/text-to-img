import math
from abc import ABCMeta
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn.utils.rnn import pad_sequence


import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import minbpe.base
import minbpe.regex
import util
from util import load_texts


class AbsTransformer(nn.Module, metaclass=ABCMeta):
    pass


class CrossAttentionTransformer(AbsTransformer):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, max_seq_length=512, dropout=0.1):
        """
        Initializes the Cross-Attention Transformer model.

        Args:
        - vocab_size (int): The size of the vocabulary.
        - d_model (int): The dimension/width of the model & embeddings (hidden size).
        - nhead (int): The number of attention heads.
        - num_encoder_layers (int): The number of encoder layers.
        - num_decoder_layers (int): The number of decoder layers.
        - dim_feedforward (int): The dimension of the feedforward network.
        - max_seq_length (int): The maximum sequence length.
        - dropout (float): The dropout rate.
        """
        super(CrossAttentionTransformer, self).__init__()

        self.d_model = d_model

        # Embedding layer for token inputs
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding to add positional information to token embedding
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Transformer decoder
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        # Output projection layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def create_positional_encoding(max_seq_length, d_model):
        """
        Creates positional encoding for sequences.

        Args:
        - max_seq_length (int): The maximum sequence length.
        - d_model (int): The dimension of the model.

        Returns:
        - pe (Parameter): Positional encoding tensor.
        """
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Defines the forward pass of the model.

        Args:
        - src (Tensor): Source sequence input.
        - tgt (Tensor): Target sequence input.
        - src_mask, tgt_mask, memory_mask: Optional masks for attention.
        - src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask: Optional padding masks.

        Returns:
        - output (Tensor): Output of the decoder projected to vocabulary size.
        """
        # Embedding and positional encoding for source sequence
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        src = src + self.positional_encoding[:src.size(0), :]
        # Pass the source sequence through the encoder
        memory = self.transformer_encoder(src, src_mask, src_key_padding_mask)

        # Embedding and positional encoding for target sequence
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = tgt + self.positional_encoding[:tgt.size(0), :]
        # Pass the target sequence and encoder memory through the decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                                          memory_key_padding_mask)

        # Project the decoder output to the vocabulary size
        output = self.fc_out(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(MyTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        causal_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(device)
        src2 = self.self_attn(src, src, src, attn_mask=causal_mask, key_padding_mask=src_key_padding_mask, is_causal=True)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MyTransformer(AbsTransformer):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([MyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, input_ids, src_key_padding_mask=None):
        # 对嵌入向量进行缩放，使得其范数与位置编码（positional encoding）的范数相当。这种缩放帮助稳定训练过程。
        input_ids = self.embedding(input_ids) * math.sqrt(self.d_model)
        input_ids = self.pos_encoder(input_ids)
        for layer in self.layers:
            input_ids = layer(input_ids, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(input_ids)


class TextDataset(Dataset):
    def __init__(self, text_list: list, tokenizer, block_size=256):
        self.examples = []

        all_text = ''
        for text in text_list:
            text = text.strip()
            # 数字5(19022) [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
            all_text += text + endoftext
            start, end = text.index('('), text.index(')')
            all_text += ("The coordinates of the %s-th %s are %s" % (text[start + 1:end], text[0:start], text[end + 1 + 1:])) + endoftext

        # Tokenize the text and convert it into token IDs
        tokenized_text = tokenizer.encode(all_text)

        # Split the tokenized text into blocks of size `block_size`
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            block = tokenized_text[i:i + block_size]
            self.examples.append(block)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Get the block and convert it to a tensor
        block = self.examples[idx]
        input_ids = torch.tensor(block[:-1], dtype=torch.long)  # All but the last token
        labels = torch.tensor(block[1:], dtype=torch.long)  # All but the first token
        return input_ids, labels


endoftext = '<|endoftext|>'

def train_tokenizer(t: minbpe.base.Tokenizer, root_dir: str, **kwargs):
    data_path = config.path(config.PathType.train, root_dir, config.transformer_train_data_file)
    texts = load_texts(data_path)
    l = len(t.vocab)
    m = {}
    for k, v in t.vocab.items():
        m[v] = True
    for line in texts:
        for c in line:
            if c.encode("utf-8") not in m:
                t.vocab[l] = c.encode("utf-8")
                m[c.encode("utf-8")] = True
                l += 1
    t.register_special_tokens({endoftext: l})
    t.save(config.path(config.PathType.model, root_dir, config.tokenizer_model_file))
    return t


def run_transformer(transformer: nn.Module, tokenizer: minbpe.base.Tokenizer, input: str):
    max_index = 0
    while max_index != tokenizer.special_tokens[endoftext]:
        # .unsqueeze(0) add a batch dimension
        output = transformer(torch.tensor(tokenizer.encode(input)).unsqueeze(0))
        # last token logits
        probabilities = F.softmax(output[0][-1], dim=0)
        max_index = torch.argmax(probabilities)
        if max_index == tokenizer.special_tokens[endoftext]:
            return input
        next = tokenizer.decode([max_index.item()])
        print(next, end='')
        input += next


def train_transformer(model: nn.Module, tokenizer: minbpe.base.Tokenizer, root_dir: str, resume: bool, epochs: int = 10, lr: float = 5e-5, batch_size: int = 2, **kwargs):
    data_path = config.path(config.PathType.train, root_dir, config.transformer_train_data_file)
    model_dir = config.directory(config.PathType.model, root_dir)

    model.train()

    if resume:
        util.resume_model(model, model_dir, 'Epoch_*_transformer_*.pth')

    texts = load_texts(data_path)
    vocab_size = len(tokenizer.vocab)
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training loop
    for epoch in range(epochs):
        for inputs, labels in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(input_ids=inputs)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Loss: {loss.item():.4f}')
        util.save_model(model, model_dir, 'Epoch_{}__transformer_{:04f}.pth'.format(epoch, loss.item()), 'Epoch_*_transformer_*.pth')

    print("Training complete.")

    # batch_size = 32
    # src_seq_length = 50
    # tgt_seq_length = 50
    #
    # # Generate random token indices for source and target sequences
    # src_input = torch.randint(0, vocab_size, (src_seq_length, batch_size))
    # tgt_input = torch.randint(0, vocab_size, (tgt_seq_length, batch_size))
    # tgt_output = torch.randint(0, vocab_size, (tgt_seq_length, batch_size))
    #
    # # Define the loss function and the optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # num_epochs = 10  # Number of epochs to train over
    #
    # model.train()  # Set the model to training mode
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()  # Clear gradients
    #
    #     # Forward pass
    #     output = model(src_input, tgt_input)
    #
    #     # Output shape will be [tgt_seq_length, batch_size, vocab_size]
    #     # Reshape to calculate loss
    #     output = output.view(-1, vocab_size)
    #     tgt_output = tgt_output.view(-1)
    #
    #     # Compute loss
    #     loss = criterion(output, tgt_output)
    #
    #     # Backward pass
    #     loss.backward()  # Compute gradients
    #
    #     # Update weights
    #     optimizer.step()
    #
    #     # Print progress
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')