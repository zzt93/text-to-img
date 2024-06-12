import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# Ensure consistent results
torch.manual_seed(42)


class CrossAttentionTransformer(nn.Module):
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


def get_vocab_size():
    return 100


if __name__ == '__main__':
    vocab_size = get_vocab_size()
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_length = 512
    dropout = 0.1

    model = CrossAttentionTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                      dim_feedforward, max_seq_length, dropout)

    batch_size = 32
    src_seq_length = 50
    tgt_seq_length = 50

    # Generate random token indices for source and target sequences
    src_input = torch.randint(0, vocab_size, (src_seq_length, batch_size))
    tgt_input = torch.randint(0, vocab_size, (tgt_seq_length, batch_size))
    tgt_output = torch.randint(0, vocab_size, (tgt_seq_length, batch_size))

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # Number of epochs to train over

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        output = model(src_input, tgt_input)

        # Output shape will be [tgt_seq_length, batch_size, vocab_size]
        # Reshape to calculate loss
        output = output.view(-1, vocab_size)
        tgt_output = tgt_output.view(-1)

        # Compute loss
        loss = criterion(output, tgt_output)

        # Backward pass
        loss.backward()  # Compute gradients

        # Update weights
        optimizer.step()

        # Print progress
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')