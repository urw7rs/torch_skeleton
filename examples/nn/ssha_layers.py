import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)

        attention = self.softmax(torch.einsum("nic,njc->nij", k, q))
        x = torch.matmul(attention, x)
        return x


class NewSSHA(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead):
        super().__init__()
        self.multi_head_attn = nn.ModuleList([Attention(d_model) for _ in range(nhead)])

        self.att_linear = nn.Linear(d_model * nhead, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):
        att_x = []
        for attn_f in self.multi_head_attn:
            att_x.append(attn_f(x))

        x = torch.cat(att_x, dim=-1)

        x = self.att_linear(x)

        x = self.ffn(x)

        return x


class SSHA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_subset=8,
        num_node=20,
        kernel_size=1,
        stride=1,
        glo_reg_s=True,
        att_s=True,
        glo_reg_t=False,
        use_temporal_att=False,
        use_spatial_att=True,
        attentiondrop=0,
        use_pes=True,
        residual=True,
    ):
        super().__init__()
        inter_channels = out_channels // num_subset
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.use_pes = use_pes
        self.window_size = 3
        self.residual = residual

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        self.out_nett = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                (7, 1),
                padding=(3, 0),
                bias=True,
                stride=(stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node * self.window_size, num_node))
            self.register_buffer("atts", atts)
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(
                    in_channels, num_subset * inter_channels, 1, bias=True
                )
                self.in_nets_upfold = nn.Conv2d(
                    in_channels, num_subset * inter_channels, 1, bias=True
                )
                self.upfold = UnfoldTemporalWindows(self.window_size, 1, 1)
                self.diff_net = nn.Conv2d(in_channels, in_channels, 1, bias=True)
                self.alphas = nn.parameter.Parameter(
                    torch.ones(1, num_subset, 1, 1), requires_grad=True
                )

            if glo_reg_s:
                self.attention0s = nn.parameter.Parameter(
                    torch.ones(1, num_subset, num_node * self.window_size, num_node)
                    / num_node,
                    requires_grad=True,
                )

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    (1, 3),
                    padding=(0, 1),
                    bias=True,
                    stride=1,
                ),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )

            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(pad, 0),
                    bias=True,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.softmax = nn.Softmax()
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)
        self.resi_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # N T V C
        x = x.permute(0, 3, 1, 2)
        identity = x
        N, C, T, V = x.size()
        # print(self.betas)
        if self.use_spatial_att:
            attention = self.atts
            y = x
            if self.att_s:
                upfold = self.upfold(y)
                k = self.in_nets(y).view(
                    N, self.num_subset, self.inter_channels, T, V
                )  # nctv -> n num_subset c'tv
                q = self.in_nets_upfold(upfold).view(
                    N, self.num_subset, self.inter_channels, T, self.window_size * V
                )

                attention = (
                    attention
                    + self.soft(
                        torch.einsum("nsctu,nsctv->nsuv", [q, k])
                        / (self.inter_channels * T)
                    )
                    * self.alphas
                )

            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            # y = torch.einsum('nctv,nsuv->nsctu', [x, attention]).contiguous().view(N, self.num_subset * self.in_channels, T, self.window_size*V)
            y = (
                torch.einsum("nctu,nsuv->nsctv", upfold, attention)
                .contiguous()
                .view(N, self.num_subset * self.in_channels, T, V)
            )

            y = self.out_nets(y)  # nctv

            # y = self.out_conv(y.view(N, self.out_channels, T, -1, V)).squeeze(3)

            y = self.relu(self.downs1(x) + y)

            y = self.ff_nets(y)

            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        if self.residual:
            z = y
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)
            z += identity
            z = self.resi_relu(z)
            z = z.permute(0, 2, 3, 1)
        else:
            z = y
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)
            z = z.permute(0, 2, 3, 1)
        return z


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (
            window_size + (window_size - 1) * (window_dilation - 1) - 1
        ) // 2
        self.unfold = nn.Unfold(
            kernel_size=(self.window_size, 1),
            dilation=(self.window_dilation, 1),
            stride=(self.window_stride, 1),
            padding=(self.padding, 0),
        )

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)  # (N, C*Window_Size, (T-Window_Size+1)*(V-1+1))
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x
