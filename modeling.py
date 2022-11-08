import torch
import torch.nn as nn
from utils import *
class MPF(nn.Module):
    def __init__(self):
        super(MPF, self).__init__()
        # self.trans = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=8, layers=3)
        self.mpl = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=1)
        self.relul = nn.ReLU()
        self.mlp1l = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.mlp2l = nn.Linear(TEXT_DIM, TEXT_DIM)

        self.mpv = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=1)
        self.reluv = nn.ReLU()
        self.mlp1v = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.mlp2v = nn.Linear(TEXT_DIM, TEXT_DIM)

        self.mpa = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=1)
        self.relua = nn.ReLU()
        self.mlp1a = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.mlp2a = nn.Linear(TEXT_DIM, TEXT_DIM)

        # self.trans = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=8, layers=3)
        self.mpc = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=1)
        self.reluc = nn.ReLU()
        self.mlp1c = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.mlp2c = nn.Linear(TEXT_DIM, TEXT_DIM)


    def forward(self, xli, xvi, xai, xci):
        xli, xvi, xai, xci = xli.transpose(0, 1), xvi.transpose(0, 1), xai.transpose(0, 1), xci.transpose(0, 1)
        xlci, attnl = self.mpl(xli, xci, xci)
        glci =  xli + xlci
        xli1 = glci + self.mlp2l(self.relul(self.mlp1l(glci)))

        xvci, attnv = self.mpv(xvi, xci, xci)
        gvci =  xvi + xvci
        xvi1 = gvci + self.mlp2v(self.reluv(self.mlp1v(gvci)))

        xaci, attna = self.mpa(xai, xci, xci)
        gaci =  xai + xaci
        xai1 = gaci + self.mlp2a(self.relua(self.mlp1a(gaci)))

        xui = torch.cat([xli, xvi, xai], dim=1)
        xcui, attncu = self.mpa(xci, xui, xui)
        gcui =  xci + xcui
        xci1 = gcui + self.mlp2c(self.reluc(self.mlp1c(gcui)))
        xli1, xvi1, xai1, xci1 = xli1.transpose(0, 1), xvi1.transpose(0, 1), xai1.transpose(0, 1), xci1.transpose(0, 1) 
        return xli1, xvi1, xai1, xci1, attnl, attnv, attna, attncu 

class TMM(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(TMM, self).__init__()
        self.fc_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.fc_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.trans_l = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=1, layers=1)
        self.trans_v = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=1, layers=1)
        self.trans_a = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=1, layers=1)
        self.trans_c = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=1, layers=1)
        self.mlp_c = nn.Linear(TEXT_DIM*3, TEXT_DIM)

        self.mpf1 = MPF()
        self.mpf2 = MPF()
        self.mpf3 = MPF()


        
        self.trans_1 = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=1, layers=1)
        self.trans_2 = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=1, layers=1)
        self.trans_3 = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=1, layers=1)
        # self.pool_1 = MAGPooler(dim=TEXT_DIM)
        # self.pool_2 = MAGPooler(dim=TEXT_DIM)
        # self.pool_3 = MAGPooler(dim=TEXT_DIM)

        self.loss_la = NTXentLoss(batch_size=BATCH_SIZE)
        self.loss_lv = NTXentLoss(batch_size=BATCH_SIZE)
        self.loss_al = NTXentLoss(batch_size=BATCH_SIZE)
        self.loss_av = NTXentLoss(batch_size=BATCH_SIZE)
        self.loss_vl = NTXentLoss(batch_size=BATCH_SIZE)
        self.loss_va = NTXentLoss(batch_size=BATCH_SIZE)
        # self.mse1 = nn.MSELoss()
        # self.mse2 = nn.MSELoss()
        # self.mse3 = nn.MSELoss()
        self.m_loss_1 = MSEModel()
        self.m_loss_2 = MSEModel()
        self.m_loss_3 = MSEModel()
    def forward(self, text_embedding, visual, acoustic, input_mask_mix=None, training=None, label_ids=None):
        # bert torch.Size([32, 50, 768]) torch.Size([32, 50, 47]) torch.Size([32, 50, 74])
        # xlnet torch.Size([50, 32, 768]) torch.Size([50, 32, 47]) torch.Size([50, 32, 74])
        # roberta torch.Size([32, 50, 768]) torch.Size([32, 50, 47]) torch.Size([32, 50, 74])
        # electra torch.Size([50, 32, 768]) torch.Size([50, 32, 47]) torch.Size([50, 32, 74])
        # text_embedding, visual, acoustic = text_embedding.transpose(0, 1), visual.transpose(0, 1), acoustic.transpose(0, 1)
        xl = text_embedding
        xv = self.fc_v(visual)
        xa = self.fc_a(acoustic)

        xl, xv, xa = xl.transpose(0, 1), xv.transpose(0, 1), xa.transpose(0, 1)
        xl = self.trans_a(xl, xl, xl)
        xv = self.trans_v(xv, xv, xv)
        xa = self.trans_a(xa, xa, xa)
        xl, xv, xa = xl.transpose(0, 1), xv.transpose(0, 1), xa.transpose(0, 1)

        # xl, xv, xa = xl.transpose(0, 1), xv.transpose(0, 1), xa.transpose(0, 1)
        loss_la = self.loss_la(xl[:,0,:], xa[:,1,:])
        loss_lv = self.loss_lv(xl[:,0,:], xv[:,1,:])
        loss_av = self.loss_av(xa[:,1,:], xv[:,1,:])
        loss_al = self.loss_la(xa[:,1,:], xl[:,0,:])
        loss_vl = self.loss_lv(xv[:,1,:], xl[:,0,:])
        loss_va = self.loss_av(xv[:,1,:], xa[:,1,:])
        # xl, xv, xa = xl.transpose(0, 1), xv.transpose(0, 1), xa.transpose(0, 1)
        
        xl_cls, xa_cls, xv_cls = xl[:, 0, :], xv[:, 1, :], xa[:, 1, :]
        
        center = torch.cat([xl, xa, xv], dim=2)
        xc = self.mlp_c(center)

        xc = xc.transpose(0, 1)
        xc = self.trans_c(xc, xc, xc)
        xc = xc.transpose(0, 1)

        xl1, xv1, xa1, xc1, attnl1, attnv1, attna1, attncu1 = self.mpf1(xl, xv, xa, xc)
        xl2, xv2, xa2, xc2, attnl2, attnv2, attna2, attncu2 = self.mpf2(xl1, xv1, xa1, xc1)
        xl3, xv3, xa3, xc3, attnl3, attnv3, attna3, attncu3 = self.mpf3(xl2, xv2, xa2, xc2)

        x1 = torch.cat([xl1, xv1, xa1], dim=1)
        x2 = torch.cat([xl2, xv2, xa2], dim=1)
        x3 = torch.cat([xl3, xv3, xa3], dim=1)
        x1, x2, x3 = x1.transpose(0, 1), x2.transpose(0, 1), x3.transpose(0, 1)
        x1 = self.trans_1(x1, x1, x1)
        x2 = self.trans_2(x2, x2, x2)
        x3 = self.trans_3(x3, x3, x3)
        x1, x2, x3 = x1.transpose(0, 1), x2.transpose(0, 1), x3.transpose(0, 1)

        m_loss_1 = self.m_loss_1(x1[:,0,:], label_ids)
        m_loss_2 = self.m_loss_2(x2[:,0,:], label_ids)
        m_loss_3 = self.m_loss_3(x3[:,0,:], label_ids)


        alpha = 3 / (1/m_loss_1 + 1/m_loss_2 + 1/m_loss_3)
        alpha_1 = alpha * m_loss_2 * m_loss_3
        alpha_2 = alpha * m_loss_3 * m_loss_1
        alpha_3 = alpha * m_loss_1 * m_loss_2
        alpha_sum = alpha_1 + alpha_2 + alpha_3

        alpha_1, alpha_2, alpha_3 = alpha_1 / alpha_sum, alpha_2/alpha_sum, alpha_3/alpha_sum

        x = alpha_1 * x1 + alpha_2 * x2 + alpha_3 * x3

        return x, loss_lv, loss_av, loss_al, loss_vl, loss_va
