class MaskedExtractor(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, features_dim: int, mask: torch.Tensor):
        super().__init__()
        # 1️⃣ input → hidden (sparse mask_in)
        mask_in = mask[:, n_in:n_in+n_hidden]  # columns restricted to hidden nodes
        self.fc_in = MaskedLinear(n_in, n_hidden, mask_in, out_act=nn.ReLU())
        
        # 2️⃣ hidden → output (sparse mask_out)
        mask_out = mask[n_in:n_in+n_hidden, n_in+n_hidden:]  # rows restricted to hidden nodes
        self.fc_out = MaskedLinear(n_hidden, features_dim, mask_out)
        
    def forward(self, x):
        h = self.fc_in(x)
        return self.fc_out(h) 