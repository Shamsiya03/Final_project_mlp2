def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
class UNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(UNet, self).__init__()
        self.inc = inconv(inp_dim, 64)
        self.down1 = down(64, 128)
        self.up4 = up(192, 128)
        self.dcs0 = DCS(128, 32, 3)
        self.outc = outconv(128, mod_dim2)
        self.dcs1 = DCS(mod_dim2, 32, 3)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up4(x2, x1)
        x = self.outc(x)
        x, mu = self.dcs1(x)
        return x