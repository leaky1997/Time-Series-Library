import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_TSMixer
from einops.layers.torch import Rearrange, Reduce
from einops import repeat,rearrange,reduce
from mega import MultiHeadedEMA

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size = 3,stride = 1, dim=2):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding = (kernel_size-1)//2 )
        self.dim = dim
    def forward(self, x):
        x = rearrange(x, 'b L1 L2 L3 c -> b c L1 L2 L3')
        x = self.avg(x)
        x = rearrange(x, 'b c L1 L2 L3 -> b L1 L2 L3 c')                                 
        return x
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1,dim=1)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        out = torch.cat((x,x),dim = -1)
        return out
class series_add(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self):
        super(series_add, self).__init__()

    def forward(self, x):
        self.res,self.moving_mean = torch.tensor_split(x, 2,dim=-1)
        return self.res + self.moving_mean
class PreNormResidual(nn.Module):
    def __init__(self, dim, kernel_size,fn):
        super().__init__()
        self.fn = fn
        # self.norm = nn.LayerNorm(dim)
        self.norm = nn.InstanceNorm3d(dim)
        self.a = nn.Parameter(torch.tensor(1.))
        self.decomp = series_decomp(kernel_size= kernel_size) # TODO hyper
    def forward(self, x):
        norm = rearrange(x, 'b L1 L2 L3 c -> b c L1 L2 L3')
        norm = self.norm(norm)  
        norm = rearrange(norm, 'b c L1 L2 L3 -> b L1 L2 L3 c')
        decomp_series = self.decomp(norm)       
        return self.fn(decomp_series) + self.a * x  # 预训练参数
    
class Linear_op(nn.Module):
    def __init__(self,dim,drop) -> None:
        super().__init__()
        assert dim %2 == 0 
        self.ema = MultiHeadedEMA
        self.Linear_time = nn.Linear(dim, dim)
        self.Linear_fre_real = nn.Linear(dim//2 + 1, dim//2 + 1)
        self.Linear_fre_imag = nn.Linear(dim//2 + 1, dim//2 + 1)
        self.drop = nn.Dropout(drop)
        self.time_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.fre_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.a = [nn.Parameter(torch.tensor(1.))]*2
        
        self.Linear_score = nn.Linear(dim, dim)
    def forward(self,x):
        
        score = reduce(x,'b (L1 h1 )  (L2 h2) (c c2) L3-> b h1 h2 c2 L3','mean',h1=1,h2=1,c2 =1)
        score = self.Linear_score(score)
        score = torch.softmax(score,dim = -1)
        
        time = self.Linear_time(x)
        
        fre = torch.fft.rfft(x,dim = -1)
        fre_real = fre.real
        fre_imag = fre.imag
        fre_real = self.Linear_fre_real(fre_real)
        fre_imag = self.Linear_fre_imag(fre_imag)
        fre = torch.fft.irfft(fre_real+1j*fre_imag,dim=-1)
        out = self.a[0]*time + self.a[1]*fre
        return self.drop(out*score)
    

    
class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)
        self.a = [nn.Parameter(torch.tensor(1.))]*3
    def forward(self, x):
        out = 0 
        for i,fn in enumerate(self.fns):
            out += self.a[i]*fn(x)
        return out #sum(map(lambda fn: fn(x), self.fns))
def model_generator(L1 = 6, L2 = 4, L3 = 4, dim = 321, depth = 4, kernel_size = 3, expansion_factor = 8, dropout = 0.1):
    c = dim * 2
    return nn.Sequential(
        Rearrange('b (L1 L2 L3) c  -> b L1 L2 L3 c ', L1 = L1, L2 =L2, L3=L3),
        *[nn.Sequential(
            PreNormResidual(dim, kernel_size, nn.Sequential(
                ParallelSum(
                    # nn.Sequential(
                    #     Rearrange('b L1 L2 L3 c -> b L1 L2 (L3 c)', L1 = L1, L2 =L2, L3=L3),
                    #     nn.Linear(c*L3, c*L3),
                    #     nn.Dropout(dropout),
                    #     Rearrange('b L1 L2 (L3 c) -> b L1 L2 L3 c', L1 = L1, L2 =L2, L3=L3),
                    # ),
                    # nn.Sequential(
                    #     Rearrange('b L1 L2 L3 c -> b L1 L3 (L2 c)', L1 = L1, L2 =L2, L3=L3),
                    #     nn.Linear(c*L2, c*L2),
                    #     nn.Dropout(dropout),
                    #     Rearrange('b L1 L3 (L2 c) -> b L1 L2 L3 c', L1 = L1, L2 =L2, L3=L3),
                    # ),
                    # nn.Sequential(
                    #     Rearrange('b L1 L2 L3 c -> b L2 L3 (L1 c)', L1 = L1, L2 =L2, L3=L3),
                    #     nn.Linear(c*L1, c*L1),
                    #     nn.Dropout(dropout),
                    #     Rearrange('b L2 L3 (L1 c) -> b L1 L2 L3 c', L1 = L1, L2 =L2, L3=L3),
                    # ),
                    
                    nn.Sequential(
                        Rearrange('b L1 L2 L3 c -> b L1 L2 c L3', L1 = L1, L2 =L2, L3=L3),
                        Linear_op(L3,dropout),
                        Rearrange('b L1 L2 c L3 -> b L1 L2 L3 c', L1 = L1, L2 =L2, L3=L3),
                    ),
                    nn.Sequential(
                        Rearrange('b L1 L2 L3 c -> b L1 L3 c L2 ', L1 = L1, L2 =L2, L3=L3),
                        Linear_op(L2,dropout),
                        Rearrange('b L1 L3 c L2 -> b L1 L2 L3 c', L1 = L1, L2 =L2, L3=L3),
                    ),
                    nn.Sequential(
                        Rearrange('b L1 L2 L3 c -> b L2 L3 c L1 ', L1 = L1, L2 =L2, L3=L3),
                        Linear_op(L1,dropout),
                        Rearrange('b L2 L3 c L1 -> b L1 L2 L3 c', L1 = L1, L2 =L2, L3=L3),
                    ),
                ),
                nn.Linear(c, c),
                series_add(),
            )),
            PreNormResidual(dim, kernel_size, nn.Sequential(
                nn.Linear(c, c * expansion_factor),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(c * expansion_factor, dim),
                nn.Dropout(dropout)
            ))            
        ) for _ in range(depth)],
        Rearrange('b L1 L2 L3 c -> b c (L1 L2 L3)', L1 = L1, L2 =L2, L3=L3),
    )

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs,L1 = 6, L2 = 4, L3 = 4, dim = 321, depth = 4, num_classes = 10, expansion_factor = 8, dropout = 0.5, individual=False): # TODO hyper
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.embedding = configs.embedding
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.dataembedding = DataEmbedding_TSMixer(d_model = dim, embed_type='fixed', freq='h', dropout=dropout)
        self.backbone =  model_generator(L1 = configs.L1,
                                         L2 = configs.L2,
                                         L3 = configs.L2,
                                         dim = configs.enc_in,
                                         depth = configs.depth,
                                         kernel_size = configs.kernel_size,
                                         expansion_factor = configs.expansion_factor,
                                         dropout = configs.dropout)
        # self.backbone =  model_generator(L1 = 6, L2 = 4, L3 = 4, dim = 321, depth = 4, num_classes = 10, expansion_factor = 8, dropout = 0., individual=False)

        if self.task_name == 'classification':
            # self.act = F.gelu
            # self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)
        if self.task_name == 'long_term_forecast':
            self.projection = nn.Linear(
                configs.seq_len, self.pred_len)
    def encoder(self, x,embedding):

        if self.embedding:
            x = self.dataembedding(x,embedding)
        x = self.backbone(x)
        out = self.projection(x)
        return out.permute(0, 2, 1)

    def forecast(self, x_enc,x_mark_enc):
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        enc_out = self.encoder(x_enc,x_mark_enc)
        
        # De-Normalization from Non-stationary Transformer
        # dec_out = enc_out * \
        #     (stdev[:, 0, :].unsqueeze(1).repeat(
        #         1, self.pred_len, 1))
        # dec_out = enc_out + \
        #     (means[:, 0, :].unsqueeze(1).repeat(
        #         1, self.pred_len, 1))
        
        return enc_out

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc,x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
