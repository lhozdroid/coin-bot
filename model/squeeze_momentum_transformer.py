import torch.nn as nn

from model.attention_pooling import AttentionPooling
from model.positional_encoding import PositionalEncoding


class SqueezeMomentumTransformer(nn.Module):
    def __init__(self, config: dict):
        """
        Initializes the Squeeze Momentum Transformer model

        Args:
            config:
        """
        super().__init__()

        # Model hyperparameters
        self.input_size = config["input_size"]
        self.seq_len = config["sequence_length"]
        self.d_model = config["model_dimension"]
        self.nhead = config["number_of_heads"]
        self.num_layers = config["number_of_layers"]
        self.ff_dim = config["feedforward_dimension"]
        self.dropout_rate = config["dropout_rate"]
        self.use_layernorm = config["use_layernorm"]
        self.classifier_hidden_layers = config["classifier_hidden_layers"]
        self.num_classes = config["number_of_classes"]
        self.initializer = config["initializer"]

        # Projects the input features to the model dimension [batch_size, seq_len, d_model]
        self.input_proj = nn.Linear(self.input_size, self.d_model)

        # Positional encoding for temporal awareness [batch_size, seq_len, d_model]
        self.positional_encoding = PositionalEncoding(self.d_model, self.seq_len)

        # Optional LayerNorm after positional encoding [batch_size, seq_len, d_model]
        self.layernorm = nn.LayerNorm(self.d_model) if self.use_layernorm else nn.Identity()

        # Transformer encoder block [batch_size, seq_len, d_model]
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.ff_dim, dropout=self.dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Attention pooling mechanism [batch_size, d_model]
        self.pooling = AttentionPooling(self.d_model)

        # Dropout layer before the classifier [batch_size, d_model]
        self.dropout = nn.Dropout(self.dropout_rate)

        # Multi-layer classifier head [batch_size, num_classes]
        classifier_layers = []
        input_dim = self.d_model
        for hidden_dim in self.classifier_hidden_layers:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            input_dim = hidden_dim
        classifier_layers.append(nn.Linear(input_dim, self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        # Initializes model weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes model weights using the configured strategy
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.initializer == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.initializer == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through the model

        Args:
            x: [batch_size, seq_len, input_size]

        Returns:
            [batch_size, num_classes]
        """
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]
        x = self.layernorm(x)  # [batch_size, seq_len, d_model]
        x = self.encoder(x)  # [batch_size, seq_len, d_model]
        x = self.pooling(x)  # [batch_size, d_model]
        x = self.dropout(x)  # [batch_size, d_model]
        x = self.classifier(x)  # [batch_size, num_classes]
        return x
