import torch
import torch.nn as nn


class SPP_Q(nn.Module):
    def __init__(self,in_ch,out_ch,down_scale,ks=3):
        super(SPP_Q, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=1, padding=ks // 2,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.Down = nn.Upsample(scale_factor=down_scale,mode="bilinear")

    def forward(self, x):
        x_d = self.Down(x)
        x_out = self.Conv(x_d)
        return x_out




class Encoder_Pos(nn.Module):
    def __init__(self, n_dims, width=32, height=32, filters=[32,64,128,256]):
        super(Encoder_Pos, self).__init__()
        print("================= Multi_Head_Encoder =================")

        self.chanel_in = n_dims
        self.rel_h = nn.Parameter(torch.randn([1, n_dims//8, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//8, 1, width]), requires_grad=True)

        self.SPP_Q_0 = SPP_Q(in_ch=filters[0],out_ch=n_dims,down_scale=1/16,ks=3)
        self.SPP_Q_1 = SPP_Q(in_ch=filters[1],out_ch=n_dims,down_scale=1/8,ks=3)
        self.SPP_Q_2 = SPP_Q(in_ch=filters[2],out_ch=n_dims,down_scale=1/4,ks=3)
        self.SPP_Q_3 = SPP_Q(in_ch=filters[3],out_ch=n_dims,down_scale=1/2,ks=3)


        self.query_conv = nn.Conv2d(in_channels = n_dims , out_channels = n_dims//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = n_dims , out_channels = n_dims//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = n_dims , out_channels = n_dims , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x,x_list):
        m_batchsize, C, width, height = x.size()
        Multi_X = self.SPP_Q_0(x_list[0]) + self.SPP_Q_1(x_list[1]) + self.SPP_Q_2(x_list[2]) + self.SPP_Q_3(x_list[3])
        proj_query = self.query_conv(Multi_X).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy_content = torch.bmm(proj_query, proj_key)


        content_position = (self.rel_h + self.rel_w).view(1, self.chanel_in//8, -1)
        content_position = torch.matmul(proj_query,content_position)
        energy = energy_content + content_position
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Decoder_Pos(nn.Module):
    def __init__(self, n_dims, width=32, height=32):
        super(Decoder_Pos, self).__init__()
        print("================= Multi_Head_Decoder =================")

        self.chanel_in = n_dims
        self.rel_h = nn.Parameter(torch.randn([1, n_dims//8, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//8, 1, width]), requires_grad=True)
        self.query_conv = nn.Conv2d(in_channels=n_dims, out_channels=n_dims // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=n_dims, out_channels=n_dims // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=n_dims, out_channels=n_dims, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,x_encoder):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x_encoder).view(m_batchsize, -1, width * height)

        energy_content = torch.bmm(proj_query, proj_key)


        content_position = (self.rel_h + self.rel_w).view(1, self.chanel_in//8, -1)
        content_position = torch.matmul(proj_query,content_position)

        energy = energy_content+content_position
        attention = self.softmax(energy)
        proj_value = self.value_conv(x_encoder).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention





class MsTNL(nn.Module):
    def __init__(self,train_dim,filters=[32,64,128,256]):
        print("============= MsTNL =============")
        super(MsTNL, self).__init__()
        self.encoder = Encoder_Pos(train_dim,width=32,height=32,filters=filters)
        self.decoder = Decoder_Pos(train_dim,width=32,height=32)

    def forward(self, x, x_list):

        x_encoder,att_en = self.encoder(x, x_list)
        x_out,att_de = self.decoder(x,x_encoder)

        return x_out
