import math
import random
import torch
import torch.nn.functional as F
from gaussian_renderer import render_fastgs
from utils.loss_utils import l1_loss

def sampling_cameras(my_viewpoint_stack, loss_history=None, ratio=0.5):
\
\
\
       
    num_cams = 10
    
    if not loss_history or len(loss_history) < num_cams:
        indices = random.sample(range(len(my_viewpoint_stack)), num_cams)
        return [my_viewpoint_stack[i] for i in indices]

    num_hard = int(num_cams * ratio)
    num_random = num_cams - num_hard
    
    sorted_views = sorted(loss_history.items(), key=lambda item: item[1], reverse=True)
    
    stack_ids = {v.image_name: i for i, v in enumerate(my_viewpoint_stack)}
    
    chosen_indices = set()
    
    for img_name, _ in sorted_views:
        if len(chosen_indices) >= num_hard:
            break
        if img_name in stack_ids:
            chosen_indices.add(stack_ids[img_name])
            
    remaining_indices = [i for i in range(len(my_viewpoint_stack)) if i not in chosen_indices]
    
    if len(remaining_indices) < num_random:
                                    
        random_selection = remaining_indices
    else:
        random_selection = random.sample(remaining_indices, num_random)
        
    chosen_indices.update(random_selection)
    
    return [my_viewpoint_stack[i] for i in chosen_indices]

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))
    return l1_loss_norm

def get_edge_mask(image_tensor):
\
\
\
\
       
    gray = image_tensor.mean(dim=0, keepdim=True).unsqueeze(0)               
    
    device = image_tensor.device
    k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float().view(1, 1, 3, 3)
    k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float().view(1, 1, 3, 3)
    
    gx = F.conv2d(gray, k_x, padding=1)
    gy = F.conv2d(gray, k_y, padding=1)
    
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
    
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
    
    return magnitude.squeeze(0)

def get_local_ssim_map(img1, img2, window_size=11, sigma=1.5):
                                                                              
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss/gauss.sum()
    gauss = gauss.to(img1.device)
    
    _1D_window = gauss.unsqueeze(1) 
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)               
    window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
    
    img1_g = img1.mean(dim=0, keepdim=True).unsqueeze(0)               
    img2_g = img2.mean(dim=0, keepdim=True).unsqueeze(0)

    mu1 = F.conv2d(img1_g, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2_g, window, padding=window_size//2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_g*img1_g, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2_g*img2_g, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1_g*img2_g, window, padding=window_size//2, groups=1) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1.0 - ssim_map.squeeze() 

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY = False):
    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        
        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
        
        if hasattr(args, 'structural_weight') and args.structural_weight > 0:
            edge_map = get_edge_mask(gt_image)                               
            ssim_error_map = get_local_ssim_map(render_image, gt_image)
            
            structure_boost = (edge_map * args.structural_weight * 0.1) +\
                              (ssim_error_map * edge_map * args.structural_weight * 0.05)
            
            combined_error = l1_loss_norm + structure_boost
                             
            metric_map = (combined_error > args.loss_thresh).int()
        else:
            combined_error = l1_loss_norm 
            metric_map = (l1_loss_norm > args.loss_thresh).int()
                                               
        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)
        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        view_error_scalar = combined_error.mean()

        if full_metric_score is None:
            full_metric_score = view_error_scalar * accum_loss_counts.clone()
        else:
            full_metric_score += view_error_scalar * accum_loss_counts
                                              
    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score) + 1e-6)
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
        
    return importance_score, pruning_score
