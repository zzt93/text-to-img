import math
from abc import ABCMeta

from torch.nn.modules.module import T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn.utils.rnn import pad_sequence


import torch
import torch.nn as nn
import torch.nn.functional as F

import attention
import config
import minbpe.base
import minbpe.regex_impl
import util
from util import load_texts


class AbsTransformer(nn.Module, metaclass=ABCMeta):

    def clear_kv_cache(self):
        pass

    def is_cache_available(self):
        return False


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
        self.encoding = torch.zeros(max_len, d_model, device=config.device)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x, index=None):
        if x.size(1) == 1:
            return x + self.encoding[:, index-1]
        else:
            return x + self.encoding[:, :x.size(1)]


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(MyTransformerDecoderLayer, self).__init__()
        self.self_attn = attention.MultiheadAttentionWithCache(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None, causal_mask=None, cache=None):
        src2, _, new_cache = self.self_attn(src, src, src, attn_mask=causal_mask, key_padding_mask=src_key_padding_mask, is_causal=True, cache=cache)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, new_cache

    def train(self: T, mode: bool = True) -> T:
        if mode is False and config.enable_kv_cache:
            self.self_attn.enable_cache()
        return super().train(mode)


class MyTransformer(AbsTransformer):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([MyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.num_layers = num_layers
        self.enable_cache = False
        self.k_caches = None
        self.v_caches = None

    def forward(self, input_ids, src_key_padding_mask=None, causal_mask=None, index=None):
        # 对嵌入向量进行缩放，使得其范数与位置编码（positional encoding）的范数相当。这种缩放帮助稳定训练过程。
        input_ids = self.embedding(input_ids) * math.sqrt(self.d_model)
        if self.enable_cache and index is not None:
            input_ids = self.pos_encoder(input_ids, index)
            # print(input_ids)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                cache = (self.k_caches[i, :index - 1, :], self.v_caches[i, :index - 1, :])
                input_ids, new_cache = layer(input_ids, src_key_padding_mask=src_key_padding_mask, causal_mask=causal_mask, cache=cache)
                # print(input_ids)
                if new_cache[0].size(0) == 1:
                    self.k_caches[i, index-1, :] = new_cache[0]
                    self.v_caches[i, index-1, :] = new_cache[1]
            if not self.training and input_ids.size(1) > 1:
                input_ids = input_ids[:, -1, :].unsqueeze(0)
            return self.decoder(input_ids)
        else:
            input_ids = self.pos_encoder(input_ids)
            # print(input_ids)
            for i in range(len(self.layers)):
                layer = self.layers[i]
                input_ids, new_cache = layer(input_ids, src_key_padding_mask=src_key_padding_mask, causal_mask=causal_mask)
                # if input_ids.size(1) > 15:
                #     print(input_ids)
                if self.enable_cache:
                    index = new_cache[0].size(0)
                    self.k_caches[i, :index, :] = new_cache[0]
                    self.v_caches[i, :index, :] = new_cache[1]
            if not self.training:
                input_ids = input_ids[:, -1, :].unsqueeze(0)
            return self.decoder(input_ids)

    def train(self: T, mode: bool = True) -> T:
        if mode is False:
            self.enable_cache = config.enable_kv_cache
            self.k_caches = torch.zeros(self.num_layers, config.max_seq_len, self.d_model, device=config.device)
            self.v_caches = torch.zeros_like(self.k_caches, device=config.device)
        return super().train(mode)


    def is_cache_available(self):
        return self.enable_cache


class PaddingTextDataset(Dataset):
    def __init__(self, text_list: list, tokenizer, block_size=256):
        self.examples = []
        self.padding_masks = []
        self.max_length = 0
        self.padding_token = 0
        token_list = []
        for text in text_list:
            text = text.strip()
            # 数字5(19022) [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
            tokenized_text = tokenizer.encode(text + endoftext)
            self.max_length = max(self.max_length, len(tokenized_text))
            token_list.append(tokenized_text)

        for tokenized_text in token_list:
            padded_text = tokenized_text + [self.padding_token] * (self.max_length - len(tokenized_text))
            self.examples.append(padded_text)

            # Create padding mask: 1 for real tokens, 0 for padding tokens
            padding_mask = [False] * len(tokenized_text) + [True] * (self.max_length - len(tokenized_text))
            self.padding_masks.append(padding_mask)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Get the block and convert it to a tensor
        block = self.examples[idx]
        input_ids = torch.tensor(block[:-1], dtype=torch.long, device=config.device)  # All but the last token
        input_mask = torch.tensor(self.padding_masks[idx][:-1], device=config.device)
        labels = torch.tensor(block[1:], dtype=torch.long, device=config.device)  # All but the first token
        return input_ids, labels, input_mask


class TextDataset(Dataset):
    def __init__(self, text_list: list, tokenizer, block_size=256):
        self.examples = []

        # all_text = ''
        # for text in text_list:
        #     text = text.strip()
        #     # 数字5(19022) [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        #     all_text += text + endoftext
        #     # start, end = text.index('('), text.index(')')
        #     # all_text += ("The coordinates of the %s-th %s are %s" % (text[start + 1:end], text[0:start], text[end + 1 + 1:])) + endoftext
        #
        # # Tokenize the text and convert it into token IDs
        # tokenized_text = tokenizer.encode(all_text)
        #
        # # Split the tokenized text into blocks of size `block_size`
        # for i in range(0, len(tokenized_text) - block_size + 1, block_size):
        #     block = tokenized_text[i:i + block_size]
        #     self.examples.append(block)

        for text in text_list:
            text = text.strip()
            # 数字5(19022) [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
            tokenized_text = tokenizer.encode(text + endoftext)
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
        return input_ids, labels, None


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


def generate_square_subsequent_mask_bool(sz: int, device) -> torch.Tensor:
    if config.enable_kv_cache:
        # F.scaled_dot_product_attention
        # 生成上三角矩阵（不包含对角线），False 表示需要遮蔽的位置
        upper_tri = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        # 转换成布尔掩码（False 表示需要遮蔽的位置）
        mask = upper_tri == 0
    else:
        # 生成上三角矩阵（不包含对角线），True 表示需要遮蔽的位置
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
    return mask


def run_transformer(transformer: AbsTransformer, tokenizer: minbpe.base.Tokenizer, input: str, force_dim: int = None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    max_index = -1
    dim = force_dim
    print(input, end='', flush=True)

    # if torch.backends.mps.is_available():
    #     # bug of MPS in pytorch 2.3.1, can't direct generate on dev. 2.6.0 is fixed
    #       causal_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(dev)
    # else:
    #
    if config.enable_kv_cache:
        mask_value = True
    else:
        mask_value = False
    full_causal_mask = torch.full(size=(1, config.max_seq_len), fill_value=mask_value, device=device, dtype=bool)
    full_2d_causal_mask = generate_square_subsequent_mask_bool(config.max_seq_len, device=device)
    # full_2d_causal_mask = nn.Transformer.generate_square_subsequent_mask(config.max_seq_len, device=device)

    index = len(tokenizer.encode(input))

    while max_index != tokenizer.special_tokens[endoftext]:
        with torch.no_grad():
            if transformer.is_cache_available() and max_index != -1:
                last = torch.tensor([max_index], device=device).unsqueeze(0)
                causal_mask = full_causal_mask[:, :index]
                # input_ids = torch.tensor(tokenizer.encode(input), device=device).unsqueeze(0)
                # causal_mask = full_2d_causal_mask[:index, :index]
                output = transformer(last, causal_mask=causal_mask, index=index)
            else:
                # .unsqueeze(0) add a batch dimension
                input_ids = torch.tensor(tokenizer.encode(input), device=device).unsqueeze(0)
                causal_mask = full_2d_causal_mask[:index, :index]
                output = transformer(input_ids, causal_mask=causal_mask)

        # if torch.cuda.is_available():
        #     with torch.no_grad():
        #         with autocast():
        #             output_fp16 = model(input_data)

        # last token logits
        probabilities = F.softmax(output[0][-1], dim=0)
        max_index = torch.argmax(probabilities)
        if max_index == tokenizer.special_tokens[endoftext]:
            return input
        next = tokenizer.decode([max_index.item()])
        if dim is not None and next == ',':
            dim = dim - 1
            if dim == 0:
                print(']\nforce end')
                input += ']'
                return input

        print(next, end='', flush=True)
        input += next
        index += 1

    transformer.clear_kv_cache()


def train_transformer(model: nn.Module, tokenizer: minbpe.base.Tokenizer, root_dir: str, resume: bool, epochs: int = 10, lr: float = 5e-5, batch_size: int = 2, **kwargs):
    data_path = config.path(config.PathType.train, root_dir, config.transformer_train_data_file)
    model_dir = config.directory(config.PathType.model, root_dir)

    model.train()

    if resume:
        util.resume_model(model, model_dir, 'Epoch_*_transformer_*.pth')

    texts = load_texts(data_path)
    vocab_size = len(tokenizer.vocab)
    dataset = PaddingTextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Training loop
    for epoch in range(epochs):
        count = 0
        for inputs, labels, padding in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
            if torch.cuda.is_available():
                print(f"CUDA Memory - Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB, "
                      f"Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
            elif torch.backends.mps.is_available():
                print(f"MPS Memory - Allocated: {torch.mps.current_allocated_memory() / (1024 ** 2):.2f} MB, "
                      f"Driver : {torch.mps.driver_allocated_memory() / (1024 ** 2):.2f} MB")

            inputs, labels, padding = inputs, labels, padding

            # Forward pass
            outputs = model(input_ids=inputs, src_key_padding_mask=padding)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Loss: {loss.item():.4f}')
            count += 1
            if count % 1000 == 999:
                util.save_model(model, model_dir, 'Epoch_{}_mid_transformer_{:04f}.pth'.format(epoch, loss.item()),
                                'Epoch_*_transformer_*.pth')

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
