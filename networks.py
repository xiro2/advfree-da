import paddle
import paddle.nn as nn
nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant())
paddle.seed(42)

class Encoder(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2D(in_channels=num_channels,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn1   = nn.BatchNorm2D(num_filters)
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn2   = nn.BatchNorm2D(num_filters)

        self.pool  = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.relu = nn.LeakyReLU()
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x_conv = x           
        x_pool = self.pool(x)
        return x_conv, x_pool
    
    
class Decoder(nn.Layer):
    def __init__(self, num_channels, num_filters,norm):
        super(Decoder,self).__init__()
        self.up1 = nn.Conv2D(in_channels=num_channels,
                                    out_channels=num_filters,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.up2 = nn.UpsamplingBilinear2D(scale_factor=2)
        if(norm=='Instance'):
            self.bn1   = nn.BatchNorm2D(num_filters)
            self.bn2   = nn.BatchNorm2D(num_filters)
        else:
            self.bn1   = nn.BatchNorm2D(num_filters)
            self.bn2   = nn.BatchNorm2D(num_filters)
        self.conv1 = nn.Conv2D(in_channels=num_filters*2,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        
        self.relu = nn.LeakyReLU()
    def forward(self,input_conv,input_pool):
        x = self.up1(input_pool)
        x = self.up2(x)

        x = paddle.concat(x=[input_conv,x],axis=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class UNet(nn.Layer):
    def __init__(self,num_classes=5,num_channels=32):
        super(UNet,self).__init__()
        self.down1 = Encoder(num_channels=  1, num_filters=num_channels) 
        self.down2 = Encoder(num_channels=num_channels, num_filters=num_channels*2)
        self.down3 = Encoder(num_channels=num_channels*2, num_filters=num_channels*4)
        self.down4 = Encoder(num_channels=num_channels*4, num_filters=num_channels*8)
        self.down5 = Encoder(num_channels=num_channels*8, num_filters=num_channels*16)
        
        self.mid_conv1 = nn.Conv2D(num_channels*16,num_channels*32,1)                 
        self.mid_bn1   = nn.BatchNorm2D(num_channels*32)
        self.relu1 = nn.LeakyReLU()
        self.mid_conv2 = nn.Conv2D(num_channels*32,num_channels*32,1)
        self.mid_bn2   = nn.BatchNorm2D(num_channels*32)
        self.relu2 = nn.LeakyReLU()

        self.up5 = Decoder(num_channels*32,num_channels*16,'Batch')            
        self.up4 = Decoder(num_channels*16,num_channels*8,'Batch')                      
        self.up3 = Decoder(num_channels*8,num_channels*4,'Batch')
        self.up2 = Decoder(num_channels*4,num_channels*2,'Batch')
        self.up1 = Decoder(num_channels*2,num_channels,'Batch')
        self.last_conv = nn.Conv2D(num_channels,num_classes,kernel_size=1)        
    def forward(self,inputs):
  
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x5, x = self.down5(x)
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.relu1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)
        x = self.relu2(x)
        x = self.up5(x5, x)
        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        x = self.last_conv(x)
        return x
