# version 1
import torch
import numpy as np
import cv2
import torch.nn as nn
import copy
import math
import os


def get_semantic_feature(rgb_image, dataset_path, feature_name):
    """Feature source switched by env var LEGO_FEATURE_MODE (set in lego_slam.py).
    online: SED+HR extracted live from RGB (768);  offline: precomputed LSeg .pt (512)."""
    if os.environ.get("LEGO_FEATURE_MODE", "online") == "online":
        from clip_sed.extractor import extract
        return extract(rgb_image)
    path = f"{dataset_path}/rgb_feature_langseg/{feature_name}"
    return torch.load(path, map_location='cpu').half()


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # C2W = torch.linalg.inv(Rt)
    # cam_center = C2W[:3, 3]
    # cam_center = (cam_center + translate) * scale
    # cam_center = (cam_center) * scale
    # C2W[:3, 3] = cam_center
    # Rt = C2W.inverse()
    return Rt

def orthonormalize_R(R):
    """
    Re-orthogonalize a 3x3 rotation matrix via SVD for numerical stability.
    """
    try:
        U, S, Vh = torch.linalg.svd(R)
        R_ortho = U @ Vh

        # Prevent reflection (det=-1) by flipping the sign of the last column
        if torch.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vh

        return R_ortho
    except torch.linalg.LinAlgError:
        print("Warning: SVD decomposition failed. Skipping orthonormalization.")
        return R

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

class SharedPoints(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.points = torch.zeros((num_points, 3)).float()
        self.colors = torch.zeros((num_points, 3)).float()
        self.z_values = torch.zeros((num_points)).float()
        self.filter = torch.zeros((num_points)).int()
        self.using_idx = torch.zeros((1)).int()
        self.filter_size = torch.zeros((1)).int()
    
    def input_values(self, new_points, new_colors, new_z_values, new_filter):
        self.using_idx[0] = new_points.shape[0]
        self.points[:self.using_idx[0],:] = new_points
        self.colors[:self.using_idx[0],:] = new_colors
        self.z_values[:self.using_idx[0]] = new_z_values
        
        self.filter_size[0] = new_filter.shape[0]
        self.filter[:self.filter_size[0]] = new_filter

    def get_values(self):
        return  copy.deepcopy(self.points[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.colors[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.z_values[:self.using_idx[0]].numpy()),\
                copy.deepcopy(self.filter[:self.filter_size[0]].numpy())

class SharedGaussians(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.xyz = torch.zeros((num_points, 3)).float().cuda()
        self.colors = torch.zeros((num_points, 3)).float().cuda()
        self.rots = torch.zeros((num_points, 4)).float().cuda()
        self.scales = torch.zeros((num_points, 3)).float().cuda()
        self.z_values = torch.zeros((num_points)).float().cuda()
        self.trackable_filter = torch.zeros((num_points)).long().cuda()
        self.using_idx = torch.zeros((1)).int().cuda()
        self.filter_size = torch.zeros((1)).int().cuda()
        self.zero_filter = torch.zeros((num_points)).long().cuda()
        self.zero_filter_size = torch.zeros((1)).int().cuda()
        self.edge_mask = torch.zeros((num_points)).long().cuda()
        self.edge_mask_size = torch.zeros((1)).int().cuda()

    def input_values(self, new_xyz, new_colors, new_rots, new_scales, new_z_values, new_trackable_filter, new_zero_filter, new_edge_mask):
        # on CPU memory
        self.using_idx[0] = new_xyz.shape[0]
        self.xyz[:self.using_idx[0],:] = new_xyz
        self.colors[:self.using_idx[0],:] = new_colors
        self.rots[:self.using_idx[0],:] = new_rots
        self.scales[:self.using_idx[0],:] = new_scales
        self.z_values[:self.using_idx[0]] = new_z_values
        
        self.filter_size[0] = new_trackable_filter.shape[0]
        self.trackable_filter[:self.filter_size[0]] = new_trackable_filter
        
        self.zero_filter_size[0] = new_zero_filter.shape[0]
        self.zero_filter[:self.zero_filter_size[0]] = new_zero_filter
        
        self.edge_mask_size[0] = new_edge_mask.shape[0]
        self.edge_mask[:self.edge_mask_size[0]] = new_edge_mask
    
    def get_values(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:]),\
                copy.deepcopy(self.colors[:self.using_idx[0],:]),\
                copy.deepcopy(self.rots[:self.using_idx[0],:]),\
                copy.deepcopy(self.scales[:self.using_idx[0],:]),\
                copy.deepcopy(self.z_values[:self.using_idx[0]]),\
                copy.deepcopy(self.trackable_filter[:self.filter_size[0]]),\
                copy.deepcopy(self.zero_filter[:self.zero_filter_size[0]]),\
                copy.deepcopy(self.edge_mask[:self.edge_mask_size[0]])

class SharedTargetPoints(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.xyz = torch.zeros((num_points, 3)).float()
        self.rots = torch.zeros((num_points, 4)).float()
        self.scales = torch.zeros((num_points, 3)).float()
        self.using_idx = torch.zeros((1)).int()
        self.edge_mask = torch.zeros((num_points)).long()

    def input_values(self, new_xyz, new_rots, new_scales, new_edge_mask):
        self.using_idx[0] = new_xyz.shape[0]
        if self.using_idx[0]>self.num_points:
            print("Too many target points")
        self.xyz[:self.using_idx[0],:] = new_xyz
        self.rots[:self.using_idx[0],:] = new_rots
        self.scales[:self.using_idx[0],:] = new_scales
        self.edge_mask[:self.using_idx[0]] = new_edge_mask
    
    def get_values_tensor(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:]),\
                copy.deepcopy(self.rots[:self.using_idx[0],:]),\
                copy.deepcopy(self.scales[:self.using_idx[0],:]),\
                copy.deepcopy(self.edge_mask[:self.using_idx[0]])

    def get_values_np(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.rots[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.scales[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.edge_mask[:self.using_idx[0]].numpy())

class SharedCam(nn.Module):
    def __init__(self, FoVx, FoVy, image, depth_image, semantic_feature,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, dataset_path=None):
        super().__init__()
        
        self.device = "cuda"
        self.cam_idx = torch.zeros((1)).int()
        self.R = torch.eye(3,3).float()
        self.t = torch.zeros((3)).float()
        self.FoVx = torch.tensor([FoVx])
        self.FoVy = torch.tensor([FoVy])
        self.image_width = torch.tensor([image.shape[1]])
        self.image_height = torch.tensor([image.shape[0]])
        self.cx = torch.tensor([cx])
        self.cy = torch.tensor([cy])
        self.fx = torch.tensor([fx])
        self.fy = torch.tensor([fy])
        self.dataset_path = dataset_path
        self.c2w = torch.eye(4,4).float()
        
        self.original_image = torch.from_numpy(image).float().permute(2,0,1)/255
        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0)
        self.semantic_feature_name_size = torch.zeros((1)).int().cuda()
        self.semantic_feature_name_size[0] = len(semantic_feature)
        self.semantic_feature_name = torch.zeros([100], dtype=torch.int32).cuda()
        self.semantic_feature_name[:self.semantic_feature_name_size[0]] = semantic_feature
        
        # Load semantic feature immediately and store it
        if dataset_path is not None:
            self.semantic_feature_image = get_semantic_feature(image, dataset_path, self.get_semantic_feature_name())
        else:
            self.semantic_feature_image = None

        # self.cam_rot_delta = nn.Parameter(
        #     torch.zeros(3, requires_grad=True, dtype=torch.float32)
        # )
        # self.cam_trans_delta = nn.Parameter(
        #     torch.zeros(3, requires_grad=True, dtype=torch.float32)
        # )
        
        # self.cam_rot_delta = torch.zeros(3, requires_grad=True, dtype=torch.float32)
        # self.cam_trans_delta = torch.zeros(3, requires_grad=True, dtype=torch.float32)

        # l = [
        #     {'params': [self.cam_rot_delta], 'lr': 0.003 * 0.5, "name": "rot_cam"},
        #     {'params': [self.cam_trans_delta], 'lr': 0.001 * 0.5, "name": "trans_cam"}
        # ]

        # self.cam_pose_optimizer = torch.optim.Adam(l)

        if self.semantic_feature_image is not None:
            self.semantic_height = torch.tensor([self.semantic_feature_image.shape[1]])
            self.semantic_width = torch.tensor([self.semantic_feature_image.shape[2]])
        else:
            self.semantic_height = torch.tensor([0])
            self.semantic_width = torch.tensor([0])
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(self.R, self.t, trans, scale).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.projection_matrix = getProjectionMatrix2(
            znear=self.znear,
            zfar=self.zfar,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.image_width,
            H=self.image_height,
        ).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def setup_optimizer(self):
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device="cuda", dtype=torch.float32)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device="cuda", dtype=torch.float32)
        )
        
        l = [
            {'params': [self.cam_rot_delta], 'lr': 0.003 * 0.5, "name": "rot_cam"},
            {'params': [self.cam_trans_delta], 'lr': 0.001 * 0.5, "name": "trans_cam"}
        ]

        self.cam_pose_optimizer = torch.optim.Adam(l)
        
    def update_pose(self):
        # print(self.cam_trans_delta, self.cam_rot_delta)
        tau = torch.cat([self.cam_trans_delta, self.cam_rot_delta], axis=0)
        # tau = torch.cat([-self.cam_trans_delta.grad*0.001 * 0.5, -self.cam_rot_delta.grad*0.003 * 0.5], axis=0)

        T_w2c = torch.eye(4, device=tau.device)
        T_w2c[0:3, 0:3] = self.R
        T_w2c[0:3, 3] = self.t
        
        # T_w2c = T_w2c.to(torch.double)
        # tau = tau.to(torch.double)

        new_w2c = SE3_exp(tau) @ T_w2c
        # new_w2c = new_w2c.to(torch.float32)
        # new_c2w = torch.linalg.inv(new_w2c)
        # new_c2w = new_w2c.inverse()

        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]

        # new_R = orthonormalize_R(new_R)

        converged = tau.norm() < 1e-4
        self.update_RT(new_R, new_T)
        
        self.update_matrix()

        with torch.no_grad():
            self.cam_rot_delta.data.fill_(0)
            self.cam_trans_delta.data.fill_(0)
            
            self.cam_trans_delta.grad.zero_()
            self.cam_rot_delta.grad.zero_()
    
    def update_RT(self, R, t):
        self.R[:,:] = R.to(device=self.device)
        self.t[:] = t.to(device=self.device)
        
        w2c = torch.eye(4,4)
        w2c[:3,:3] = self.R
        w2c[:3,3] = self.t
        self.c2w[:,:] = w2c.inverse()
    
    def update_matrix(self):
        self.world_view_transform[:,:] = getWorld2View2(self.R, self.t, self.trans, self.scale).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform[:,:] = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center[:] = self.world_view_transform.inverse()[3, :3]
    
    def setup_cam(self, R, t, rgb_img, depth_img, semantic_feature):
        # Set pose, projection matrix
        c2w = torch.eye(4,4)
        c2w[:3,:3] = torch.from_numpy(R)
        c2w[:3,3] = torch.from_numpy(t)
        self.c2w[:,:] = c2w
        
        w2c = c2w.inverse()
        
        self.R[:,:] = w2c[:3,:3]
        self.t[:] = w2c[:3,3]
        
        self.update_matrix()
        
        # Update image
        self.original_image[:,:,:] = torch.from_numpy(rgb_img).float().permute(2,0,1)/255
        self.original_depth_image[:,:,:] = torch.from_numpy(depth_img).float().unsqueeze(0)
        self.semantic_feature_name_size[0] = len(semantic_feature)
        self.semantic_feature_name[:self.semantic_feature_name_size[0]] = semantic_feature
        

        if self.dataset_path is not None:
            self.semantic_feature_image[:,:,:] = get_semantic_feature(rgb_img, self.dataset_path, self.get_semantic_feature_name())


    def get_semantic_feature_name(self):
        semantic_feature_name = ''.join([chr(c.item()) for c in self.semantic_feature_name[:self.semantic_feature_name_size[0]]])
        return semantic_feature_name


    def get_semantic_feature_cuda(self):
        """Get semantic feature moved to CUDA device"""
        if self.semantic_feature_image is not None:
            return self.semantic_feature_image.cuda()
        else:
            return None
    
    def on_cuda(self):
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.full_proj_transform = self.full_proj_transform.cuda()
        self.camera_center = self.camera_center.cuda()
        
        self.R = self.R.cuda()
        self.t = self.t.cuda()
        
        self.original_image = self.original_image.cuda()
        self.original_depth_image = self.original_depth_image.cuda()
        # Pin memory for efficient transfer to GPU when needed
        if self.semantic_feature_image is not None:
            self.semantic_feature_image = self.semantic_feature_image.pin_memory()
            
        self.cam_rot_delta = torch.zeros(3, requires_grad=True, device="cuda", dtype=torch.float32)
        self.cam_trans_delta = torch.zeros(3, requires_grad=True, device="cuda", dtype=torch.float32)
            
    

class MappingCam(nn.Module):
    def __init__(self, cam_idx, R, t, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 semantic_feature_name=None, dataset_path=None
                 ):
        super().__init__()
        self.cam_idx = cam_idx
        self.R = R
        self.t = t
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.last_loss = 0.
        self.dataset_path = dataset_path
        
        self.original_image = torch.from_numpy(image).float().cuda().permute(2,0,1)/255
        # rgb_level_1 = cv2.resize(image, (self.image_width//2, self.image_height//2))
        # rgb_level_2 = cv2.resize(image, (self.image_width//4, self.image_height//4))
        # self.rgb_level_1 = torch.from_numpy(rgb_level_1).float().cuda().permute(2,0,1)/255
        # self.rgb_level_2 = torch.from_numpy(rgb_level_2).float().cuda().permute(2,0,1)/255
        
        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0).cuda()
        # depth_level_1 = cv2.resize(depth_image, (self.image_width//2, self.image_height//2), interpolation=cv2.INTER_NEAREST)
        # depth_level_2 = cv2.resize(depth_image, (self.image_width//4, self.image_height//4), interpolation=cv2.INTER_NEAREST)
        # self.depth_level_1 = torch.from_numpy(depth_level_1).float().unsqueeze(0).cuda()
        # self.depth_level_2 = torch.from_numpy(depth_level_2).float().unsqueeze(0).cuda()
        
        # Load semantic feature if available
        # self.semantic_feature_image = None
        # if semantic_feature_name and dataset_path:
        #     try:
        #         semantic_feature_path = f"{dataset_path}/rgb_feature_langseg/{semantic_feature_name}"
        #         self.semantic_feature_image = torch.load(semantic_feature_path, map_location='cpu').half().pin_memory()
        #     except (FileNotFoundError, OSError) as e:
        #         print(f"Warning: Semantic feature file not found: {semantic_feature_path}")
        #         self.semantic_feature_image = None
        self.semantic_feature_image = None
        if semantic_feature_name and dataset_path:
            try:
                self.semantic_feature_image = get_semantic_feature(image, dataset_path, semantic_feature_name).pin_memory()
            except:
                self.semantic_feature_image = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def get_semantic_feature_cuda(self):
        """Get semantic feature moved to CUDA device"""
        if self.semantic_feature_image is not None:
            return self.semantic_feature_image.cuda()
        else:
            return None
            
    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.t, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class CNN_encoder(nn.Module):
    def __init__(self, input_dim, output_dim, enable_training=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, kernel_size=1).cuda()
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1).cuda()
        self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1).cuda()
        self.relu = nn.ReLU()

        # Only create optimizer if training is enabled
        if enable_training:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        else:
            self.optimizer = None

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
        
        

class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=1).cuda()
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1).cuda()
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        # Store inverse matrices for each layer
        self.conv1_invmatrix = torch.linalg.pinv(self.conv1.weight.data.squeeze(-1).squeeze(-1))
        self.conv2_invmatrix = torch.linalg.pinv(self.conv2.weight.data.squeeze(-1).squeeze(-1))

    def forward(self, x):
        # input_dim -> 128
        x = self.conv1(x)
        x = self.relu(x)
        
        # 128 -> output_dim
        x = self.conv2(x)
        
        return x

    def inv_conv(self, y):
        with torch.no_grad():
            # Inverse of conv2: output_dim -> 128
            x = torch.matmul(y, self.conv2_invmatrix.T)
            x = self.relu(x)
            
            # Inverse of conv1: 128 -> input_dim
            x = torch.matmul(x, self.conv1_invmatrix.T)
            
            return x
        
    def set_invmatrix(self):
        self.conv1_invmatrix[:,:] = torch.linalg.pinv(self.conv1.weight.data.squeeze(-1).squeeze(-1))
        self.conv2_invmatrix[:,:] = torch.linalg.pinv(self.conv2.weight.data.squeeze(-1).squeeze(-1))
        return


