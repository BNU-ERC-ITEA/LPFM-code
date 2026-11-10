import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

total_imw = 320
total_imh = 256
# ----------- RoPE 2D Implementation -----------
def apply_rope_2d(x, H, W):
    """
    Safe 2D RoPE
    x: [B, N, Hh, L, D]  (batch, window_num, heads, seq_len, dim_per_head)
    H, W: feature map height and width (token map size after window flattening)
    """
    B, N, Hh, L, D = x.shape
    d_half = D // 2
    d1 = d_half
    d2 = D - d_half  # support odd D

    # Generate position indices
    pos_h = torch.arange(H, device=x.device)
    pos_w = torch.arange(W, device=x.device)
    grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
    grid = torch.stack([grid_h, grid_w], dim=-1).reshape(-1, 2)  # [L,2]

    # Frequency
    theta1 = 10000 ** (torch.arange(d1, device=x.device).float() / d1)
    theta2 = 10000 ** (torch.arange(d2, device=x.device).float() / d2)

    h_emb1 = grid[:, 0][:, None] / theta1  # [L, d1]
    w_emb1 = grid[:, 1][:, None] / theta1
    h_emb2 = grid[:, 0][:, None] / theta2  # [L, d2]
    w_emb2 = grid[:, 1][:, None] / theta2

    # sin/cos
    sin1, cos1 = torch.cat([h_emb1.sin(), w_emb1.sin()], dim=-1), torch.cat([h_emb1.cos(), w_emb1.cos()], dim=-1)
    sin2, cos2 = torch.cat([h_emb2.sin(), w_emb2.sin()], dim=-1), torch.cat([h_emb2.cos(), w_emb2.cos()], dim=-1)

    # Crop to D
    sin = torch.cat([sin1, sin2], dim=-1)[..., :D]
    cos = torch.cat([cos1, cos2], dim=-1)[..., :D]

    # Expand to batch/head
    sin = sin[None, None, None, :, :].to(x.dtype)
    cos = cos[None, None, None, :, :].to(x.dtype)

    # Rotate
    d_half = D // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    x_rot = torch.cat([x1 * cos[..., :d_half] - x2 * sin[..., :d_half],
                       x1 * sin[..., :d_half] + x2 * cos[..., :d_half]], dim=-1)
    return x_rot



# ----------- HS_MSA with RoPE (non-local) -----------
class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            only_local_branch=False,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        # --- Position encoding for local branch ---
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads // 2, seq_l1, seq_l1))
            trunc_normal_(self.pos_emb1)
            # non-local branch no longer needs pos_emb2, directly use RoPE

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, \
            'fmap dimensions must be divisible by the window size'

        if self.only_local_branch:
            # ----- local only -----
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c',
                              b0=w_size[0], b1=w_size[1])
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c',
                            h=h // w_size[0], w=w // w_size[1], b0=w_size[0])
        else:
            q = self.to_q(x)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q1, q2 = q[:, :, :, :c // 2], q[:, :, :, c // 2:]
            k1, k2 = k[:, :, :, :c // 2], k[:, :, :, c // 2:]
            v1, v2 = v[:, :, :, :c // 2], v[:, :, :, c // 2:]

            # ---- local branch ----
            q1, k1, v1 = map(lambda t: rearrange(
                t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(
                t, 'b n mm (h d) -> b n h mm d', h=self.heads // 2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # ---- non-local branch with RoPE ----
            q2, k2, v2 = map(lambda t: rearrange(
                t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3),
                             (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(
                t, 'b n mm (h d) -> b n h mm d', h=self.heads // 2), (q2, k2, v2))

            # RoPE injects position information
            q2 = apply_rope_2d(q2, h // w_size[0], w // w_size[1])
            k2 = apply_rope_2d(k2, h // w_size[0], w // w_size[1])

            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            # ---- merge ----
            out = torch.cat([out1, out2], dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        return out

class HSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class HST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1,1,1]):
        super(HST, self).__init__()
        self.dim = dim
        self.out_dim=out_dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                HSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)
        x = x[:,:self.out_dim,:,:]

        # Encoder
        fea_encoder = []
        for (HSAB, FeaDownSample) in self.encoder_layers:
            fea = HSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, HSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = HSAB(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]



def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs
def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class FReLU(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride=1, **kwargs):
        super(FReLU, self).__init__()
        self.depth = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2, groups=input_ch)

    def forward(self, x):
        depth = self.depth(x)
        return torch.max(x, depth)


class ReLU(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__()
        pass

    def forward(self, x):
        return torch.relu(x)


class Leaky(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Leaky, self).__init__()
        self.alpha = kwargs.get('alpha', .02)

    def forward(self, x):
        return torch.where(x < 0., x * self.alpha, x)


class Swish(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()
        pass

    def forward(self, x):
        return swish(x)


class Mish(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Mish, self).__init__()
        pass

    def forward(self, x):
        return mish(x)

class Base_Module(torch.nn.Module):

    def __init__(self):
        super(Base_Module, self).__init__()

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            return x
class HSI_prior_block(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, activation='relu'):
        super(HSI_prior_block, self).__init__()
        self.activation = activation
        self.spatial_1 = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.spectral = torch.nn.Conv2d(output_ch, input_ch, 1, 1, 0)
        self.embedding = nn.Conv2d(input_ch*2, input_ch, 3, 1, 1, bias=False)

    def forward(self, x,mask):
        x_in = self.embedding(torch.cat([x, mask], dim=1))
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)

        return x


class Net(nn.Module):

    def __init__(self, bands=30):
        super(Net, self).__init__()

        self.num_iterations = 9
        self.denoisers = nn.ModuleList([])
        self.init_denoiser(bands, self.num_iterations)

        self.eps = nn.Parameter(torch.zeros(self.num_iterations))  # Step size ε
        self.eta = nn.Parameter(torch.zeros(self.num_iterations))  # Penalty term η
    def init_denoiser(self, in_channels, num_iterations):
        for _ in range(num_iterations):
            self.denoisers.append(
                HSI_prior_block(input_ch=in_channels, output_ch=in_channels),
            )





    def forward(self, y, input_mask=None):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        Phi, Phi_s = input_mask
        b,c,h,w = Phi.shape



        def A(x, Phi):
            temp = x * Phi
            y = torch.sum(temp, 1).unsqueeze(1)
            return y

        def At(y, Phi):
            temp = y.repeat(1, Phi.shape[1], 1, 1)
            x = temp * Phi
            return x
        A_ = lambda x,Phi: A(self.shift(x), Phi)
        At_ = lambda y,Phi: self.shift_back(At(y, Phi))

        MASK = self.shift_back(Phi)
        z0 = At_(y, Phi)
        z = At_(y,Phi)
        for i in range(self.num_iterations):
            x = z + At_(torch.div(y - A_(z,Phi), self.eps[i] + Phi_s), Phi)
            z = self.denoisers[i](x, MASK)

        return z

