import torch


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=1,
                                hidden_size=hidden_size,
                                batch_first=True)

    def forward(self, x: torch.Tensor):
        batch_size, seq_length, num_features = x.shape
        initial_hidden_state = torch.zeros((1, batch_size, self.hidden_size))
        _, context = self.rnn(x, initial_hidden_state)

        return context


class Decoder(torch.nn.Module):
    def __init__(self, input_size, seq_length):
        super(Decoder, self).__init__()

        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=seq_length,
                                batch_first=False)

    def forward(self, context: torch.Tensor):
        output, _ = self.rnn(context)

        return output.permute(1, 2, 0)


class Autoencoder(torch.nn.Module):
    def __init__(self, hidden_size=3, seq_length=7):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(hidden_size=hidden_size)
        self.decoder = Decoder(input_size=hidden_size, seq_length=seq_length)

    def forward(self, x: torch.Tensor):
        context = self.encoder(x)
        output = self.decoder(context)

        return output


if __name__ == "__main__":
    sample = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
    ]).reshape((2, 7, 1)).type(torch.FloatTensor)

    encoder = Encoder()
    decoder = Decoder()

    context = encoder.forward(sample)
    decoder.forward(context)
