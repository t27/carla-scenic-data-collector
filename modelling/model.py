import torch
import torch.nn as nn

# input data size N,10,25,25


class ScenarioModel(nn.Module):
    def __init__(self, input_channels=10, input_dim=25, hidden_dim=1024) -> None:
        super(type(self), self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, out_channels=16, kernel_size=3, padding=1
            ),  # 25x25
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),  # 13x13
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),  # 13x13
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),  # 7x7
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=128,
                out_channels=hidden_dim,
                kernel_size=7,
                stride=1,
                padding=0,
            ),  # N x hidden_dim x 1 x 1
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dim,
                out_channels=128,
                kernel_size=7,
                stride=1,
                padding=0,
            ),  # N x hidden_dim x 1 x 1
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1
            ),  # 7x7
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, padding=1
            ),  # 13x13
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1
            ),  # 13x13
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=input_channels, kernel_size=3, padding=1
            ),  # 25x25
        )

    def embedding(self, x):
        return self.encoder(x)

    def forward(self, x, sigmoid=False):
        emb = self.encoder(x)
        out = self.decoder(emb)
        if sigmoid:
            out = torch.sigmoid(out)
        return out


if __name__ == "__main__":
    model = ScenarioModel(input_channels=10, input_dim=25, hidden_dim=1024)
    test = torch.ones(6, 10, 25, 25)
    model(test)

