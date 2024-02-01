import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def double_conv(in_ch, out_ch, kernel_size, stride_layer_1, stride_layer_2,activation=nn.ReLU):

    # Define a double convolution block with two 3x3 convolutional layers
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride_layer_1),
        activation(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride_layer_2),
        activation(inplace=True),
    )
    return conv

def disp_conv(in_ch, out_ch, kernel_size, stride_layer_1, stride_layer_2,activation=nn.ReLU):

    # Define a double convolution block with two 3x3 convolutional layers
    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride_layer_1),
        activation(inplace=True),
    )
    return conv

def upconv(in_ch, out_ch, kernel_size, stride_layer_1):

    conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride_layer_1),
        nn.ReLU(inplace=True),
        )
    return conv

def get_disp(x):
        disp = disp_conv(x, 2, 3, 1,nn.Sigmoid)
        return disp

def pad_tensor(original_tensor, target_tensor):
   
    layer1_shape = original_tensor.size()
    layer2_shape = target_tensor.size()
    padding_top = (layer2_shape[2] - layer1_shape[2]) // 2
    padding_bottom = layer2_shape[2] - layer1_shape[2] - padding_top
    padding_left = (layer2_shape[3] - layer1_shape[3]) // 2
    padding_right = layer2_shape[3] - layer1_shape[3] - padding_left

    # Apply zero padding to layer1 to match the size of layer2
    # The padding has the order (padding_left, padding_right, padding_top, padding_bottom)
    original_padded = F.pad(original_tensor, (padding_left, padding_right, padding_top, padding_bottom), 'constant', 0)
    return original_padded

class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super(UNet, self).__init__()

        # Define the U-Net architecture with encoder and decoder blocks
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(in_channels, 32, 7, 2, 1)
        self.down_conv_2 = double_conv(32, 64, 5, 2, 1)
        self.down_conv_3 = double_conv(64, 128, 3, 2, 1)
        self.down_conv_4 = double_conv(128, 256, 3, 2, 1)
        self.down_conv_5 = double_conv(256, 512, 3, 2, 1)
        self.down_conv_6 = double_conv(512, 512, 3, 2, 1)

        self.up_trans_6 = nn.ConvTranspose2d(
            in_channels=512, 
            out_channels=512,
            kernel_size=3, 
            stride=3)
        
        self.iconv6 = upconv(1024, 512,3,1)

        self.up_trans_5 = nn.ConvTranspose2d(
            in_channels=512, 
            out_channels=256,
            kernel_size=3, 
            stride=2)
        
        self.iconv5= upconv(512, 256,3,1)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=256, 
            out_channels=128,
            kernel_size=3, 
            stride=2)
        
        self.iconv4 = upconv(256, 128,3,1)
        self.disp4_trans_conv = nn.ConvTranspose2d(
            in_channels=2, 
            out_channels=2,
            kernel_size=3, 
            stride=2,padding = 1,output_padding=1)
        self.disp4 = get_disp(128)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64,
            kernel_size=3, 
            stride=2)
        
        self.iconv3= upconv(130, 64,3,1)
        self.disp3_trans_conv = nn.ConvTranspose2d(
            in_channels=2, 
            out_channels=2,
            kernel_size=3, 
            stride=2,padding = 1,output_padding=1)
        self.disp3 = get_disp(64)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=64, 
            out_channels=32,
            kernel_size=3, 
            stride=1)
        
        self.iconv2= upconv(66, 32,3,1)
        self.disp2_trans_conv = nn.ConvTranspose2d(
            in_channels=2, 
            out_channels=2,
            kernel_size=3, 
            stride=2,padding = 1,output_padding=1)
        self.disp2 = get_disp(32)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=32, 
            out_channels=16,
            kernel_size=3, 
            stride=2)
        
        self.iconv1= upconv(18, 16,3,1)
       
        self.disp1 = get_disp(16)


        # Sigmoid activation to constrain output values between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):

        # U-Net forward pass with encoder and decoder
        # input_image: (640x480x3)
        x1 = self.down_conv_1(image)
        # x1: (311x231x32)        
        x2 = self.down_conv_2(x1)
        # x2: (160x110x64)       
        x3 = self.down_conv_3(x2)
        # x3: (72x52x128)            
        x4 = self.down_conv_4(x3)
        # x4: (33x23x256)            
        x5 = self.down_conv_5(x4)
        # x5: (14x9x512)
        x6 = self.down_conv_6(x5)
        # x6: (4x2x512)

        ######Skip Connections#####
        skip1 = x5   #conv5
        skip2 = x4   #conv4
        skip3 = x3   #conv3
        skip4_1= x2  #conv2
        skip5_1 = x1 #conv1 


        # Decoder with transposed convolutions
        x = self.up_trans_6(x6)
        y = pad_tensor(x,skip1)
        # y: (14x9x512)
        x = self.iconv6(torch.cat([y, skip1], 1))
        # x: (12x7x512)

        x= self.up_trans_5(x)
        y = pad_tensor(x,skip2)
        # y: (33x23x256)
        x = self.iconv5(torch.cat([y, skip2], 1))
        # x: (31x21x256)

        x = self.up_trans_4(x)
        y = pad_tensor(x,skip3)
        # y: (72x52x128)
        x= self.iconv4(torch.cat([y, skip3], 1))
        # x: (70x50x128)
        dispare4 =self.disp4(y)
        # dispare4: (70x50x2)
        x = self.up_trans_3(x)
        y = pad_tensor(x,skip4_1)
        # y: (150x110x64)

        ### UPSAMPLE dipare4 *2
        dispare4 = self.disp4_trans_conv(dispare4)

        # dispare4: (140x100x2)
        intermediate_disparity_1 = pad_tensor(dispare4,skip4_1)
        #print("CHECK before Concating")
        #print(skip4_1.size())
        #print(y.size())
        #print(disparity.size())
        x = self.iconv3(torch.cat([y,intermediate_disparity_1,skip4_1], 1))
        #print("After CONV:",x.size())
        dispare3 = self.disp3(x)
        #print("______________________________")
        #print("Disparity Layer Dispare 3:",dispare3.size())
        #print("______________________________")

        x = self.up_trans_2(x)
        y = pad_tensor(x,skip5_1)
        #print("Trans_Conv_5_with padding",y.size())

        ### UPSAMPLE dipare3 *2
        dispare3 = self.disp3_trans_conv(dispare3)
        #print("Trans_Conv_disparity4",dispare3.size())
        intermediate_disparity_2 = pad_tensor(dispare3,skip5_1)
        
        #print("CHECK before Concating")
        #print(skip5_1.size())
        #print(y.size())
        #print(disparity.size())

        x = self.iconv2(torch.cat([y,intermediate_disparity_2,skip5_1], 1))
        dispare2 = self.disp2(x)
        #print("After CONV:",x.size())
        #print("______________________________")
        #print("Disparity Layer Dispare 2:",dispare2.size())
        #print("______________________________")

        x = self.up_trans_1(x)

        ### UPSAMPLE dipare2 *2
        #print("Trans_Conv_6_with padding",x.size())
        dispare2 = self.disp2_trans_conv(dispare2)
        intermediate_disparity_3 = pad_tensor(dispare2,x)
        #print("Check before concating")
        #print(x.size())
        #print(disparity.size())

        x = self.iconv1(torch.cat([x,intermediate_disparity_3], 1))
        #print("After Conv:",x.size())
        dispare1 = self.disp1(x)

        # print("Disparity Layer FINAL:",dispare1.size())
        #print("final output",x.size())
        final_disparity = dispare1

        return final_disparity,intermediate_disparity_3, intermediate_disparity_2, intermediate_disparity_1
    

# if __name__==  "__main__":
#     model = UNet()
#     x = torch.randn([1, 3, 640, 480])
#     final_disparity, intermediate_disparity_1, intermediate_disparity_2, intermediate_disparity_3 = model(x)
#     print(final_disparity.size(), intermediate_disparity_1.size(), intermediate_disparity_2.size(), intermediate_disparity_3.size())
