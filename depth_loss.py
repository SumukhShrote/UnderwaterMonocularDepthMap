import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F

def bilinear_sampler_1d_h(input_image, disparity, wrap_mode='border', tensor_type = 'torch.cuda.FloatTensor'):
    device = 'cuda'
    num_batch, num_channels, height, width = input_image.size()

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        input_image = pad(input_image, (1, 1, 1, 1))
    elif wrap_mode == 'edge':
        edge_size = 0
    else:
        return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_image = input_image.permute(1, 0, 2, 3).contiguous()
    im_flat = input_image.view(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type)
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type)
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.reshape(-1).repeat(1, num_batch)
    y = y.reshape(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    x = x + disparity.contiguous().view(-1) * width
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type)
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.view(num_channels, num_batch, height, width).permute(1,0,2,3)

    return output

def scale_pyramid(input_image):
    # Generates image pyramid 
    scaled_images = []
    scaled_images.append(F.interpolate(input_image, size=(615, 455), mode='area'))
    scaled_images.append(F.interpolate(input_image, size=(619, 459), mode='area'))
    scaled_images.append( F.interpolate(input_image, size=(311, 231), mode='area'))
    scaled_images.append( F.interpolate(input_image, size=(150, 110), mode='area'))
    return scaled_images


def gradient_x(img):
    # Padding to the right to maintain original width
    img_padded = F.pad(img, (0, 1, 0, 0))
    gx = img_padded[:,:, :, :-1] - img_padded[:, :, :, 1:]
    return gx

def gradient_y(img):
    # Padding to the bottom to maintain original height
    img_padded = F.pad(img, (0, 0, 0, 1))
    gy = img_padded[:,:, :-1, :] - img_padded[ :,:, 1:, :]
    return gy

def recompute_weights(pyramid): 
    revised_weights_x = []
    revised_weights_y = []

    for img in pyramid:
        # compute image gradients in x and y directions
        img_gradients_x = gradient_x(img)
        img_gradients_y = gradient_y(img)

        # Compute weights based on the mean absolute gradients
        weights_x = torch.exp(-torch.mean(torch.abs(img_gradients_x), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(img_gradients_y), dim=1, keepdim=True))

        revised_weights_x.append(weights_x)
        revised_weights_y.append(weights_y)

    return revised_weights_x, revised_weights_y

def disparity_smoothness_loss(disparities, scale_pyramid):
    # Compute gradients of disparities in x and y directions
    disp_gradients_x = [gradient_x(d) for d in disparities]
    disp_gradients_y = [gradient_y(d) for d in disparities]

    # Recompute weights for each level in the scale pyramid
    weights_x, weights_y = recompute_weights(scale_pyramid)

    # Apply computed weights to disparity gradients
    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

    # Combine gradients and weights for x and y directions
    tensor_list = [smoothness_x[i] + smoothness_y[i] for i in range(4)]

    # Calculate the mean absolute value of the combined gradients
    smooth_loss_list = [torch.mean(abs(x)) for x in tensor_list]

    # Calculate the overall smoothness loss
    return torch.tensor([torch.mean(torch.tensor(smooth_loss_list))])

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute local means of input images
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=0)  

    # Compute local variances of input images
    sigma_x = F.avg_pool2d(x * 2, kernel_size=3, stride=1, padding=0) - mu_x * 2
    sigma_y = F.avg_pool2d(y * 2, kernel_size=3, stride=1, padding=0) - mu_y * 2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=0) - mu_x * mu_y

    # Calculate SSIM components
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x * 2 + mu_y * 2 + C1) * (sigma_x + sigma_y + C2)

    # Compute SSIM index
    SSIM = SSIM_n / SSIM_d

    # Clamp and calculate SSIM loss
    x = torch.clamp((1 - SSIM) / 2, 0, 1)
    loss= torch.mean(x)
    
    return loss

def reconstruction_loss(scaled_image_1, scaled_image_2, disparity):
    
    #Construct right image using left image and right disparity
    reconstructed_image = bilinear_sampler_1d_h(scaled_image_1, disparity)
    
    #L1 loss
    loss = nn.L1Loss()
    l1_loss = loss(reconstructed_image, scaled_image_2)

    #SSIM Loss
    ssim_loss = SSIM(reconstructed_image, scaled_image_2)

    # Weighted Sum
    l1_loss = 0.15 * l1_loss
    ssim_loss = 0.85 * ssim_loss

    recons_loss = torch.add(l1_loss, ssim_loss)
    return recons_loss

def depth_loss(left_image, right_image, disparity_1, disparity_2, disparity_3, disparity_4):
    # Calculate scale pyramids for input images 
    left_scale_pyramid  = scale_pyramid(left_image)
    right_scale_pyramid = scale_pyramid(right_image)

    # Creating left and right disparity arrays
    left_disp = [disparity_1[:, 0:1, :, :], disparity_2[:, 0:1, :, :], disparity_3[:, 0:1, :, :], disparity_4[:, 0:1, :, :]]
    right_disp = [disparity_1[:, 1:, :, :], disparity_2[:, 1:, :, :], disparity_3[:, 1:, :, :], disparity_4[:, 1: , :, :]]

    #Reconstruction Loss
    recon_loss_right = [reconstruction_loss(left_scale_pyramid[i],right_scale_pyramid[i],right_disp[i]) for i in range(4)]
    recon_loss_left = [reconstruction_loss(right_scale_pyramid[i],left_scale_pyramid[i],left_disp[i]) for i in range(4)]
    total_recon_loss = [recon_loss_left[i]+recon_loss_right[i] for i in range(4)]
    total_recon_loss_avg = torch.mean(torch.stack(total_recon_loss))

    #Disparity Smoothness Loss
    disp_smooth_loss_left = disparity_smoothness_loss(left_disp, left_scale_pyramid)
    disp_smooth_loss_right = disparity_smoothness_loss(right_disp, right_scale_pyramid)
    total_disp_smooth_loss_avg = torch.add(disp_smooth_loss_right,disp_smooth_loss_left).to('cuda')
    
    # Total Loss
    total_loss = total_recon_loss_avg + 0.1*total_disp_smooth_loss_avg[0]
    return total_loss
    


# if __name__ == '__main__':
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # print("Using device:", device)

#     # input_image = torch.randn([1, 3, 640, 480]).to(device)

#     # disp =[torch.randn([1, 2, 615, 455]).to(device), torch.randn([1, 2, 619, 459]).to(device), torch.randn([1, 2, 311, 231]).to(device), torch.randn([1, 2, 150, 110]).to(device)]

#     # pyramid = scale_pyramid(input_image)    

#     # dep_loss = depth_loss(input_image,input_image,disp[0],disp[1],disp[2],disp[3])
#     # print(dep_loss)