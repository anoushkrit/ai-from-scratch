import torch 
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerModel(nn.Module):

    def __init__(self, n_input, n_head, n_hidden, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(n_input, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.n_input = n_input
        
        ## Add other layers
        self.decoder = nn.Linear(n_input, 1)
        self.lin = nn.Linear(40, 11)
        self.softmax = nn.Softmax(dim=1)
        # self.init_weights()

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = output.squeeze()
        output = self.softmax(self.lin(output))

        return output
