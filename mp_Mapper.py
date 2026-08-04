import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import copy
import random
import sys
import cv2
import numpy as np
import time
import rerun as rr
import pickle
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.loss_utils import l1_loss, ssim, cos_loss
import encoding.utils as utils
from encoding.models.sseg import BaseNet
from scene import GaussianModel
from gaussian_renderer import render, render_3, network_gui, render_3_filtered
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from Lseg.modules.lseg_module import LSegModule
import open3d as o3d
import torch.nn.functional as F
import torch.nn as nn
import clip
from PIL import Image
import torchvision
import sklearn
from mp_Loopmanager import LoopManager
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation
from evo.core import sync
from evo.tools import plot
import evo.main_ape as main_ape
from scipy.spatial.transform import Rotation

from Lseg.modules.models.lseg_vit import _make_pretrained_clip_vitl16_384

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

class Mapper(SLAMParameters):
    def __init__(self, slam):   
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.save_results = slam.save_results
        self.save_image_flag = slam.save_image_flag  # [enable, interval]
        self.rerun_viewer = slam.rerun_viewer
        self.speedup = slam.speedup
        self.semantic_feature_init = slam.semantic_feature_init
        self.max_mapping_keyframes = slam.max_mapping_keyframes
        self.post_training_iter = slam.post_training_iter
        self.eval_ratio = slam.eval_ratio
        self.dist_threshold = slam.dist_threshold
        self.sim_threshold = slam.sim_threshold
        self.max_sample_size = slam.max_sample_size
        self.sample_ratio = slam.sample_ratio
        self.k_nearest = slam.k_nearest
        self.prune_min_opacity = slam.prune_min_opacity
        self.pretrained_encoder_path = slam.pretrained_encoder_path
        self.pretrained_decoder_path = slam.pretrained_decoder_path
        self.encoder_flag = slam.encoder_flag
        self.encoder_warmup_iter = slam.encoder_warmup_iter
        self.encoder_train_interval = slam.encoder_train_interval
        self.encoder_train_duration = slam.encoder_train_duration
        self.loopclosing_global_correspondence_distance = slam.loopclosing_global_correspondence_distance
        self.loopclosing_local_correspondence_distance = slam.loopclosing_local_correspondence_distance
        self.loop_constraint_noise = slam.loop_constraint_noise
        
        # Loop closing configuration from main SLAM system
        self.enable_loop_closing = getattr(slam, 'enable_loop_closing', False)
        if self.enable_loop_closing:
            print("Loop Closing: ENABLED (Language-based)")
            # Initialize histograms dictionary for loop closure
            self.histograms = {}

            # Load language codebook for histogram generation (mode-aware: online=768 SED, offline=512 LSeg)
            self.codebook_path = f"saved/language_codebook_{os.environ.get('LEGO_FEATURE_MODE', 'online')}.pkl"
            self.vocabulary = None
            self.vocabulary_gpu = None  # Cache for GPU vocabulary
            self.num_clusters = 64
            self.load_codebook()
        else:
            print("Loop Closing: DISABLED")
            self.histograms = None
            self.vocabulary = None
            self.vocabulary_gpu = None

        self.iter_shared = slam.iter_shared
        self.semantic_feature_dim = slam.semantic_feature_dim
        self.point_feature_dim = slam.point_feature_dim

        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])

        # Distortion coefficients for camera undistortion
        self.k1 = float(self.camera_parameters[9])
        self.k2 = float(self.camera_parameters[10])
        self.p1 = float(self.camera_parameters[11])
        self.p2 = float(self.camera_parameters[12])
        self.k3 = float(self.camera_parameters[13])

        self.dist_coeffs = np.array(
            [
                self.k1,
                self.k2,
                self.p1,
                self.p2,
                self.k3,
            ]
        )

        # Initialize undistortion maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.cam_intrinsic,
            self.dist_coeffs,
            np.eye(3),
            self.cam_intrinsic,
            (self.W, self.H),
            cv2.CV_32FC1,
        )

        self.downsample_rate = slam.downsample_rate
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.activate_rendering_loss_refinement = True
        
        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        self.opacity_threshold = 0.95
        
        if self.trajmanager.which_dataset == "replica":
            self.prune_th = 2.5
        else:
            self.prune_th = 10.0
        if getattr(slam, "prune_th_cli", -1.0) > 0:
            self.prune_th = slam.prune_th_cli
        
        
        
        self.downsample_idxs, self.x_pre, self.y_pre, self.downsample_idxs_sparse, self.x_pre_sparse, self.y_pre_sparse = self.set_downsample_filter_multires(self.downsample_rate)

        self.gaussians = GaussianModel(self.sh_degree, self.point_feature_dim)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
        self.mapping_losses = []
        self.new_keyframes = []

        # Initialize ATE metrics
        self.ate_rmse = None
        self.ate_mean = None
        self.ate_max = None
        self.ate_min = None
        self.ate_median = None
        self.ate_sse = None
        self.ate_std = None
        
        self.all_kf_poses = []      # c2w
        self.all_kf_poses_idxs = [] # for trajectory evaluation
        self.tracking_kf_idxs = []  # in the all_kf_poses list
        self.mapping_kf_idxs = []
        self.n_trackable_keyframes = slam.n_trackable_keyframes
        self.loop_closing_start = slam.loop_closing_start
        self.pose_lr_rate = slam.pose_lr_rate
        
        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
        self.refined_pose_shared = slam.refined_pose_shared
        self.from_last_loop_closed = 100
        
        if self.speedup:
            self.cnn_encoder = slam.cnn_encoder

            self.cnn_decoder = slam.cnn_decoder

            if self.pretrained_encoder_path:
                device = next(self.cnn_encoder.parameters()).device
                ckpt = torch.load(self.pretrained_encoder_path, map_location=device)
                state_dict = ckpt["model_state_dict"]
                self.cnn_encoder.load_state_dict(state_dict)
                print(f"Loaded pretrained encoder")

            if self.pretrained_decoder_path:
                device = next(self.cnn_decoder.parameters()).device
                ckpt = torch.load(self.pretrained_decoder_path, map_location=device)
                state_dict = ckpt["model_state_dict"]
                self.cnn_decoder.load_state_dict(state_dict)
                print(f"Loaded pretrained decoder")
        else:
            self.cnn_decoder = None
        
        # Initialize LoopManager only if loop closing is enabled
        if self.enable_loop_closing:
            self.loop_manager = LoopManager(self)
        else:
            self.loop_manager = None

    def is_encoder_period(self, train_iter=None):
        """Check if current iteration is in an encoder training period."""
        if train_iter is None:
            train_iter = self.train_iter
        if train_iter < self.encoder_warmup_iter:
            return False
        elapsed = (train_iter - self.encoder_warmup_iter) % self.encoder_train_interval
        return elapsed < self.encoder_train_duration

    def run(self):
        self.mapping()
    
    def set_downsample_filter_multires( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        sample_interval_sparse = downsample_scale*2
        
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        self.u_downsampled = u
        self.v_downsampled = v
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        # sparse
        h_val_sparse = sample_interval_sparse * torch.arange(0,int(self.H/sample_interval_sparse)+1)
        h_val_sparse = h_val_sparse-1
        h_val_sparse[0] = 0
        h_val_sparse = h_val_sparse*self.W
        a_sparse, b_sparse = torch.meshgrid(h_val_sparse, torch.arange(0,self.W,sample_interval_sparse))
        # For tensor indexing, we need tuple
        pick_idxs_sparse = ((a_sparse+b_sparse).flatten(),)
        # Get u, v values
        v_sparse, u_sparse = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u_sparse = u_sparse.flatten()[pick_idxs_sparse]
        v_sparse = v_sparse.flatten()[pick_idxs_sparse]
        
        self.u_downsampled_sparse = u_sparse
        self.v_downsampled_sparse = v_sparse
        
        # Calculate xy values, not multiplied with z_values
        x_pre_sparse = (u_sparse-self.cx)/self.fx # * z_values
        y_pre_sparse = (v_sparse-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre, pick_idxs_sparse, x_pre_sparse, y_pre_sparse
    
    def load_codebook(self):
        """Load language codebook for histogram generation"""
        if not os.path.exists(self.codebook_path):
            print(f"Warning: Codebook file not found: {self.codebook_path}")
            print("   Loop closing will be disabled")
            self.enable_loop_closing = False
            return
        
        try:
            with open(self.codebook_path, 'rb') as f:
                codebook_data = pickle.load(f)
            
            self.vocabulary = codebook_data['vocabulary']
            self.num_clusters = codebook_data['num_clusters']
            
            print(f"Codebook loaded: {self.num_clusters} clusters x {self.vocabulary.shape[1]} dimensions")
        except Exception as e:
            print(f"Error loading codebook: {e}")
            self.vocabulary = None
            self.enable_loop_closing = False
    
    def assign_pixels_to_codebook(self, feature_tensor):
        """
        Assign each pixel to the most similar codebook entry using cosine similarity (GPU optimized)
        
        Args:
            feature_tensor: [512, H, W] torch tensor (on GPU)
        
        Returns:
            assignments: [H, W] numpy array (codebook indices)
        """
        if self.vocabulary is None:
            return None
        
        # Convert Float16 → Float32 if needed
        if feature_tensor.dtype == torch.float16:
            feature_tensor = feature_tensor.float()
        
        # Reshape: [512, H, W] → [H*W, 512] (keep on GPU)
        C, H, W = feature_tensor.shape
        pixels_gpu = feature_tensor.permute(1, 2, 0).reshape(-1, C)  # [H*W, 512], stays on GPU
        
        # Move vocabulary to GPU (cache it for efficiency)
        if self.vocabulary_gpu is None:
            self.vocabulary_gpu = torch.from_numpy(self.vocabulary).float().cuda()  # [64, 512]
        
        # Calculate cosine similarity on GPU (much faster!)
        pixels_normalized = F.normalize(pixels_gpu, dim=1)  # [H*W, 512]
        vocab_normalized = F.normalize(self.vocabulary_gpu, dim=1)  # [64, 512]
        
        similarities = torch.mm(pixels_normalized, vocab_normalized.t())  # [H*W, 64]
        
        # Find most similar codebook (highest similarity)
        assignments = torch.argmax(similarities, dim=1)  # [H*W]
        assignments = assignments.reshape(H, W).cpu().numpy()  # Only final result goes to CPU
        
        return assignments
    
    def create_histogram(self, assignments):
        """
        Create histogram from assignments
        
        Args:
            assignments: [H, W] numpy array
        
        Returns:
            histogram: [64] numpy array (normalized)
        """
        if assignments is None:
            return None
            
        # Calculate histogram
        histogram, _ = np.histogram(assignments.flatten(), 
                                 bins=self.num_clusters, 
                                 range=(0, self.num_clusters))
        
        # Normalize (convert to probability)
        histogram = histogram.astype(np.float32)
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def log_module_norms(self, module, name, prefix):
        """Check model training"""
        param_vec = nn.utils.parameters_to_vector(module.parameters())
        param_norm = param_vec.norm().item()
        print(f" {name} {prefix}-step param L2 norm: {param_norm:.4f}")
        if any(p.grad is not None for p in module.parameters()):
            grad_vec = torch.cat([p.grad.flatten() for p in module.parameters() if p.grad is not None])
            grad_norm = grad_vec.norm().item()
            print(f" {name} {prefix}-step grad  L2 norm: {grad_norm:.4f}")  

    def generate_and_store_histogram(self, kf_idx):
        """
        Generate histogram for current keyframe and store it (performance optimized)
        
        Args:
            kf_idx: keyframe index
        """
        if not self.enable_loop_closing or self.vocabulary is None:
            return
            
        try:
            # Get semantic feature from current keyframe
            gt_semantic_feature = self.shared_cam.get_semantic_feature_cuda()
            if gt_semantic_feature is None:
                return
            
            # Assign pixels to codebook
            assignments = self.assign_pixels_to_codebook(gt_semantic_feature)
            if assignments is None:
                return
            
            # Create histogram
            histogram = self.create_histogram(assignments)
            if histogram is None:
                return
            
            # Store in dictionary
            self.histograms[kf_idx] = histogram
            
        except Exception as e:
            print(f"Error generating histogram for keyframe {kf_idx}: {e}")
    


    def mapping(self):
        t = torch.zeros((1,1)).float().cuda()
        if self.verbose:
            network_gui.init("127.0.0.1", 6009)
        
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()
        
        # Mapping Process is ready to receive first frame
        self.is_mapping_process_started[0] = 1
        
        # Wait for initial gaussians
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-15)
        
        self.total_start_time_viewer = time.time()
        
        points, colors, rots, scales, z_values, trackable_filter, zero_filter, edge_mask = self.shared_new_gaussians.get_values()
        
        if self.semantic_feature_init:
            # Use preloaded semantic feature from SharedCam instead of loading from disk
            gt_semantic_feature = self.shared_cam.get_semantic_feature_cuda()
            semantic_feature_img = F.interpolate(gt_semantic_feature.float().unsqueeze(0), size=(self.H, self.W), mode='nearest').squeeze() 
            
            semantic_feature_point = self.downsample_feature(semantic_feature_img.permute(1, 2, 0), (zero_filter,))
            del gt_semantic_feature, semantic_feature_img
            if self.speedup:
                feat = semantic_feature_point.unsqueeze(-1).unsqueeze(-1)   # [N, 512, 1, 1]
                out = self.cnn_encoder(feat)                                # [N, output_dim, 1, 1]
                semantic_feature_point = out.squeeze(-1).squeeze(-1)        # [N, output_dim]
        else:
            semantic_feature_point = None

        newcam = copy.deepcopy(self.shared_cam)
        newcam.on_cuda()
        newcam.keyframe_type = 'tracking'  # This is a tracking keyframe

        self.all_kf_poses.append(newcam.c2w.numpy())
        self.tracking_kf_idxs.append(len(self.all_kf_poses)-1)
        self.all_kf_poses_idxs.append(newcam.cam_idx[0])
        
        # Generate histogram for initial keyframe
        if self.enable_loop_closing:
            self.generate_and_store_histogram(len(self.all_kf_poses)-1)
        
        self.gaussians.create_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter, semantic_feature_point, edge_mask, self.speedup)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        self.mapping_cams.append(newcam)
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams)-1)
        
        self.is_tracking_keyframe_shared[0] = 0

        new_keyframe = False

        while True:
            try_loop_closing = False

            if self.end_of_dataset[0]:
                # Record the iteration count when dataset ends
                if not hasattr(self, 'dataset_end_iter'):
                    self.dataset_end_iter = self.train_iter
                # Break after post_training_iter additional iterations
                if self.train_iter - self.dataset_end_iter > self.post_training_iter:
                    break
 
            if self.verbose:
                self.run_viewer()
            
            if self.is_tracking_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, trackable_filter, zero_filter, edge_mask = self.shared_new_gaussians.get_values()
                
                if self.semantic_feature_init:
                    # Use preloaded semantic feature from SharedCam instead of loading from disk
                    gt_semantic_feature = self.shared_cam.get_semantic_feature_cuda()
                    semantic_feature_img = F.interpolate(gt_semantic_feature.float().unsqueeze(0), size=(self.H, self.W), mode='nearest').squeeze() 
                    
                    semantic_feature_point = self.downsample_feature(semantic_feature_img.permute(1, 2, 0), (zero_filter,))
                    del gt_semantic_feature, semantic_feature_img
                    if self.speedup:
                        feat = semantic_feature_point.unsqueeze(-1).unsqueeze(-1)   # [N, 512, 1, 1]
                        out = self.cnn_encoder(feat)                                # [N, output_dim, 1, 1]
                        semantic_feature_point = out.squeeze(-1).squeeze(-1)        # [N, output_dim]
                else:
                    semantic_feature_point = None

                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                newcam.keyframe_type = 'tracking'  # This is a tracking keyframe

                self.all_kf_poses.append(newcam.c2w.numpy())
                self.tracking_kf_idxs.append(len(self.all_kf_poses)-1)
                self.all_kf_poses_idxs.append(newcam.cam_idx[0])
                
                # Generate histogram for new tracking keyframe
                current_kf_idx = len(self.all_kf_poses) - 1
                if self.enable_loop_closing:
                    self.generate_and_store_histogram(current_kf_idx)
            
                # rendering loss refinement
                gt_image = newcam.original_image.cuda()
                vis_img = gt_image.detach().cpu().numpy().transpose(1,2,0)
                vis_img = np.clip(vis_img, 0., 1.0) * 255
                rr.log("refinement/gt_img", rr.Image(vis_img))
                
                with torch.no_grad():
                    newcam.cam_rot_delta.data.fill_(0)
                    newcam.cam_trans_delta.data.fill_(0)
                
                l = [
                    {'params': [newcam.cam_rot_delta], 'lr': 0.003 * self.pose_lr_rate, "name": "rot_cam"},
                    {'params': [newcam.cam_trans_delta], 'lr': 0.001 * self.pose_lr_rate, "name": "trans_cam"}
                ]
                
                pose_optimizer = torch.optim.Adam(l)
                
                if self.activate_rendering_loss_refinement:
                    for refine_iter in range(10):
                        
                        pose_optimizer.zero_grad()
                        gt_image = newcam.original_image
                        gt_depth_image = newcam.original_depth_image
                        
                        _, h, w = gt_image.shape
                        mask_shape = (1, h, w)
                        
                        render_pkg = render_3(newcam, self.gaussians, self.pipe, self.background)
                        depth_image = render_pkg["render_depth"]
                        rendered_alpha = render_pkg["alpha"]
                        image = render_pkg["render"]
                        
                        mask = (gt_depth_image>0.01).view(*mask_shape)
                        mask = mask.detach()
                        
                        alpha_mask = (rendered_alpha > self.opacity_threshold).view(*mask_shape)
                        rgb_pixel_mask = (gt_image.sum(dim=0) > 0.01).view(*mask_shape)
                        
                        depth_filter = gt_depth_image < 5.0
                        depth_residual = torch.abs(depth_image - gt_depth_image)
                        depth_residual = depth_residual * alpha_mask * depth_filter * mask
                        distance_filter = depth_residual < 0.05
                        
                        color_residual = torch.abs(image - gt_image).mean(0)
                        color_filter = color_residual < 0.1 * rgb_pixel_mask
                        
                        total_filter = mask * distance_filter * color_filter
                        valid_pro = total_filter.sum() / mask.sum()
                        
                        if (valid_pro.item() < 0.3 or depth_residual.mean().item() > 0.1):
                            break
                        
                        masked_image = image * total_filter.detach()
                        masked_gt_image = gt_image * total_filter.detach()
                        masked_depth_image = depth_image * total_filter.detach()
                        masked_gt_depth_image = gt_depth_image * total_filter.detach()
                        
                        
                        Ll1_map, Ll1 = l1_loss(masked_image, masked_gt_image)
                        Ll1_d_map, Ll1_d = l1_loss(masked_depth_image, masked_gt_depth_image)

                        
                        loss = Ll1 + 0.1 * Ll1_d
                        
                        loss.backward()
                        
                        
                        with torch.no_grad():
                            pose_optimizer.step()
                            
                            
                            newcam.update_pose()
                            
                            pose_optimizer.zero_grad()
                        
                            if refine_iter == 0 or refine_iter == 9:
                                vis_img = Ll1_map.detach().cpu().numpy().transpose(1,2,0)
                                vis_img = np.clip(vis_img, 0., 1.0) * 255
                                rr.log("refinement/errormap", rr.Image(vis_img))
                                
                                vis_img = masked_image.detach().cpu().numpy().transpose(1,2,0)
                                vis_img = np.clip(vis_img, 0., 1.0) * 255
                                rr.log("refinement/rendered", rr.Image(vis_img))
                
                    self.all_kf_poses[-1] = newcam.c2w.detach().cpu().numpy()
            
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
            
                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter, semantic_feature_point, 
                                                    keyframe_idx = len(self.all_kf_poses)-1, edge_mask = edge_mask,
                                                    speedup = self.speedup)

                if self.enable_loop_closing and len(self.tracking_kf_idxs) > self.loop_closing_start:
                    if len(self.tracking_kf_idxs) > self.n_trackable_keyframes:
                        cut_idx = self.tracking_kf_idxs[-self.n_trackable_keyframes]
                    else:
                        cut_idx = 0
                    
                    try_loop_closing = True
                else:
                    cut_idx = 0
                
            
                target_points, target_rots, target_scales, target_mask, target_kf_idxs  = self.gaussians.get_gaussians_tensor()

                self.from_last_loop_closed = self.from_last_loop_closed + 1

                if try_loop_closing and self.enable_loop_closing and self.loop_manager is not None and self.from_last_loop_closed > 3:
                    # get all trackable gaussians

                    target_points, target_rots, target_scales, target_mask, target_kf_idxs  = self.gaussians.get_gaussians_tensor()
                    self.loop_manager.input_gs_map(target_points, target_rots, target_scales, target_mask, target_kf_idxs)

                    # update all_kf_poses (after BA)

                    self.loop_manager.insert_new_kf(newcam.c2w.detach().cpu().numpy(), newcam.original_depth_image, len(self.all_kf_poses)-1, self.all_kf_poses, self.tracking_kf_idxs)
                    self.loop_manager.loop_detection_naive()

                    self.loop_manager.calculate_loop_constraints()

                    loop_closed = self.loop_manager.pgo_update()

                    if loop_closed:
                        self.update_gs_mapper(self.loop_manager.kf_poses_after_pgo)
                        self.from_last_loop_closed = 0

                # Visualize Gaussian map in rerun (after loop closing if it happened)
                if self.rerun_viewer:
                    points_vis = self.gaussians.get_xyz[::5].detach().cpu().numpy()
                    colors_vis = self.gaussians._features_dc[::5, 0, :].detach().cpu().numpy()
                    colors_vis = np.clip(colors_vis * 0.2821 + 0.5, 0.0, 1.0)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                    rr.log("pt/gaussian_map", rr.Points3D(points_vis, colors=colors_vis, radii=0.02))
                
                # get trackable gaussians of recent n keyframes
                target_points, target_rots, target_scales, target_kf_idxs, target_edge_mask  = self.gaussians.get_trackable_gaussians_tensor(self.trackable_opacity_th, cut_idx)
                self.shared_target_gaussians.input_values(target_points, target_rots, target_scales, target_edge_mask)
                
                # send refined pose (with loop closing or rendering-loss refinement) to tracker
                self.refined_pose_shared[:,:] = self.mapping_cams[-1].c2w
                
                self.target_gaussians_ready[0] = 1
                
                self.is_tracking_keyframe_shared[0] = 0
                
                del semantic_feature_point

            elif self.is_mapping_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, _, zero_filter, edge_mask = self.shared_new_gaussians.get_values()
                
                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                newcam.keyframe_type = 'mapping'  # This is a mapping keyframe

                ## rendering loss refinement
                
                
                self.all_kf_poses.append(newcam.c2w.numpy())
                self.all_kf_poses_idxs.append(newcam.cam_idx[0])
                
                # rendering-loss refinement
                with torch.no_grad():
                    newcam.cam_rot_delta.data.fill_(0)
                    newcam.cam_trans_delta.data.fill_(0)
                
                l = [
                    {'params': [newcam.cam_rot_delta], 'lr': 0.003 * self.pose_lr_rate, "name": "rot_cam"},
                    {'params': [newcam.cam_trans_delta], 'lr': 0.001 * self.pose_lr_rate, "name": "trans_cam"}
                ]
                
                pose_optimizer = torch.optim.Adam(l)
                
                if self.activate_rendering_loss_refinement:
                    for refine_iter in range(10):
                        
                        pose_optimizer.zero_grad()
                        gt_image = newcam.original_image
                        gt_depth_image = newcam.original_depth_image
                        
                        _, h, w = gt_image.shape
                        mask_shape = (1, h, w)
                        
                        render_pkg = render_3(newcam, self.gaussians, self.pipe, self.background)
                        depth_image = render_pkg["render_depth"]
                        rendered_alpha = render_pkg["alpha"]
                        image = render_pkg["render"]
                        
                        mask = (gt_depth_image>0.01).view(*mask_shape)
                        mask = mask.detach()
                        
                        alpha_mask = (rendered_alpha > self.opacity_threshold).view(*mask_shape)
                        rgb_pixel_mask = (gt_image.sum(dim=0) > 0.01).view(*mask_shape)
                        
                        depth_filter = gt_depth_image < 5.0
                        depth_residual = torch.abs(depth_image - gt_depth_image)
                        depth_residual = depth_residual * alpha_mask * depth_filter * mask
                        distance_filter = depth_residual < 0.05
                        
                        color_residual = torch.abs(image - gt_image).mean(0)
                        color_filter = color_residual < 0.1 * rgb_pixel_mask
                        
                        total_filter = mask * distance_filter * color_filter
                        valid_pro = total_filter.sum() / mask.sum()
                        
                        if (valid_pro.item() < 0.3 or depth_residual.mean().item() > 0.1):
                            break
                        
                        masked_image = image * total_filter.detach()
                        masked_gt_image = gt_image * total_filter.detach()
                        masked_depth_image = depth_image * total_filter.detach()
                        masked_gt_depth_image = gt_depth_image * total_filter.detach()
                        
                        Ll1_map, Ll1 = l1_loss(masked_image, masked_gt_image)
                        Ll1_d_map, Ll1_d = l1_loss(masked_depth_image, masked_gt_depth_image)
                        loss = Ll1 + 0.1 * Ll1_d
                        
                        loss.backward()
                        
                        with torch.no_grad():
                            pose_optimizer.step()

                            newcam.update_pose()
                            
                            pose_optimizer.zero_grad()
                        
                            if refine_iter == 0 or refine_iter == 9:
                                vis_img = Ll1_map.detach().cpu().numpy().transpose(1,2,0)
                                vis_img = np.clip(vis_img, 0., 1.0) * 255
                                rr.log("refinement/errormap", rr.Image(vis_img))
                                
                                vis_img = masked_image.detach().cpu().numpy().transpose(1,2,0)
                                vis_img = np.clip(vis_img, 0., 1.0) * 255
                                rr.log("refinement/rendered", rr.Image(vis_img))
                    
                    self.all_kf_poses[-1] = newcam.c2w.detach().cpu().numpy()
                
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                
                if self.semantic_feature_init:
                    # Use preloaded semantic feature from SharedCam instead of loading from disk
                    gt_semantic_feature = self.shared_cam.get_semantic_feature_cuda()
                    semantic_feature_img = F.interpolate(gt_semantic_feature.float().unsqueeze(0), size=(self.H, self.W), mode='nearest').squeeze() 
                    
                    semantic_feature_point = self.downsample_feature(semantic_feature_img.permute(1, 2, 0), (zero_filter,))
                    del gt_semantic_feature, semantic_feature_img
                    if self.speedup:
                        feat = semantic_feature_point.unsqueeze(-1).unsqueeze(-1)   # [N, 512, 1, 1]
                        out = self.cnn_encoder(feat)                                # [N, output_dim, 1, 1]
                        semantic_feature_point = out.squeeze(-1).squeeze(-1)        # [N, output_dim]
                else:
                    semantic_feature_point = None

                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor(points, colors, rots, scales, z_values, [], semantic_feature_point, 
                                                    keyframe_idx = len(self.all_kf_poses)-1, edge_mask = edge_mask,
                                                    speedup = self.speedup)
                
                self.refined_pose_shared[:,:] = self.mapping_cams[-1].c2w
                
                self.is_mapping_keyframe_shared[0] = 0
                
                del semantic_feature_point
        
            is_erased = False
            if len(self.mapping_cams)>0:
                
                # train once on new keyframe, and random
                if len(self.new_keyframes) > 0:
                    train_idx = self.new_keyframes.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True
                elif len(self.mapping_cams) > self.max_mapping_keyframes:
                    # Smart keyframe removal logic
                    # 1. Try to remove mapping keyframes first
                    mapping_indices = [i for i, cam in enumerate(self.mapping_cams) if cam.keyframe_type == 'mapping']

                    if mapping_indices:
                        # Remove oldest mapping keyframe
                        remove_idx = mapping_indices[0]
                        removed_cam = self.mapping_cams.pop(remove_idx)
                    else:
                        # No mapping keyframes left, remove from older half of tracking keyframes randomly
                        half_count = len(self.mapping_cams) // 2
                        remove_idx = random.randint(0, max(0, half_count - 1))
                        removed_cam = self.mapping_cams.pop(remove_idx)

                    train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]
                    is_erased = True
                else:
                    train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]
                
                l = [
                    {'params': [viewpoint_cam.cam_rot_delta], 'lr': 0.003 * self.pose_lr_rate, "name": "rot_cam"},
                    {'params': [viewpoint_cam.cam_trans_delta], 'lr': 0.001 * self.pose_lr_rate, "name": "trans_cam"}
                ]
                
                pose_optimizer = torch.optim.Adam(l)
                pose_optimizer.zero_grad()
                
                gt_image = viewpoint_cam.original_image.cuda()
                gt_depth_image = viewpoint_cam.original_depth_image.cuda()
                # Use preloaded semantic feature instead of loading from disk
                gt_semantic_feature = viewpoint_cam.get_semantic_feature_cuda()

                self.training=True
                render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                
                depth_image = render_pkg["render_depth"]
                image = render_pkg["render"]
                semantic_feature = render_pkg["semantic_feature"]
                
                mask = (gt_depth_image>0.)
                mask = mask.detach()
                gt_image = gt_image * mask
                
                ## feature interpolation
                semantic_feature = F.interpolate(semantic_feature.unsqueeze(0), size=(gt_semantic_feature.shape[1], gt_semantic_feature.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
                # Store original 16D feature for encoder-decoder training
                if self.speedup:
                    semantic_feature_16d = semantic_feature.clone()

                    if self.encoder_flag == 1:
                        if not self.is_encoder_period():
                            # Only decode to 512D when not in encoder-only periods
                            semantic_feature = self.cnn_decoder(semantic_feature.unsqueeze(0)).squeeze(0)  # Decode to 512D
                    else:
                        # encoder_flag=0: always decode to 512D (no encoder training)
                        semantic_feature = self.cnn_decoder(semantic_feature.unsqueeze(0)).squeeze(0)  # Decode to 512D
                
                Ll1_map, Ll1 = l1_loss(image, gt_image)
                L_ssim_map, L_ssim = ssim(image, gt_image)

                Ll1_d_map, Ll1_d = l1_loss(depth_image, gt_depth_image)

                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
                loss_d = Ll1_d
                
                # Feature loss calculation based on encoder_flag
                if self.speedup:
                    if self.encoder_flag == 1:
                        if self.train_iter < self.encoder_warmup_iter:
                            # Warming up: only decoder learning
                            # Type 1: 512D decoded feature vs GT (decoder learning) - weight 1.0
                            Ll1_feature_512d_map, Ll1_feature_512d = l1_loss(semantic_feature, gt_semantic_feature)
                            Ll1_feature = Ll1_feature_512d
                            loss = loss_rgb + 0.1*loss_d + Ll1_feature
                        elif self.is_encoder_period():
                            # Encoder-only periods
                            # Type 2: 16D encoded GT vs rendered 16D (encoder learning) - weight 1.0
                            # Convert GT to float32 and batch processing: [512,H,W] -> [1,512,H,W] -> [1,16,H,W] -> [16,H,W]
                            gt_batch = gt_semantic_feature.float().unsqueeze(0)  # [1, 512, H, W] (convert to float32)
                            gt_encoded_batch = self.cnn_encoder(gt_batch)        # Single conv operation!
                            gt_semantic_encoded = gt_encoded_batch.squeeze(0)    # [16, H, W] (float32)
                            Ll1_feature_16d_map, Ll1_feature_16d = l1_loss(semantic_feature_16d, gt_semantic_encoded)
                            loss = Ll1_feature_16d
                        else:
                            # Decoder periods: RGB + Depth + Decoder learning
                            # Type 1: 512D decoded feature vs GT (decoder learning) - weight 1.0
                            # semantic_feature is already decoded to 512D in non-encoder periods
                            Ll1_feature_512d_map, Ll1_feature_512d = l1_loss(semantic_feature, gt_semantic_feature)
                            Ll1_feature = Ll1_feature_512d
                            loss = loss_rgb + 0.1*loss_d + Ll1_feature
                    else:
                        # Original speedup mode: only decoder learning 
                        Ll1_feature_map, Ll1_feature = l1_loss(semantic_feature, gt_semantic_feature)
                        loss = loss_rgb + 0.1*loss_d + Ll1_feature
                else:
                    # Original loss for non-speedup mode
                    Ll1_feature_map, Ll1_feature = l1_loss(semantic_feature, gt_semantic_feature)
                    loss = loss_rgb + 0.1*loss_d + Ll1_feature

                del semantic_feature, gt_semantic_feature

                loss.backward()
                
                with torch.no_grad():
                    rendered_alpha = render_pkg["alpha"]
                    alpha_mask = rendered_alpha > self.opacity_threshold
                    depth_filter = gt_depth_image < 5.0
                    filtered_distance = Ll1_d_map * mask * alpha_mask * depth_filter
                    distance_filter = filtered_distance < 0.05
                    total_filter = mask * alpha_mask * distance_filter
                    valid_pro = total_filter.sum() / mask.sum()
                
                    if self.train_iter % 200 == 0:
                        self.gaussians.prune_large_transparent_and_lang(self.prune_min_opacity, self.prune_th, self.dist_threshold, self.sim_threshold, self.max_sample_size, self.sample_ratio, self.k_nearest)

                    if self.train_iter < self.encoder_warmup_iter or not self.is_encoder_period():
                        # Warming up OR Decoder periods: Update Gaussians with RGB+Depth+Decoder loss
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)
                    if len(self.mapping_cams) > 3 and valid_pro.item() > 0.5 and filtered_distance.mean().item() < 0.1 and self.activate_rendering_loss_refinement:
                        pose_optimizer.step()
                        viewpoint_cam.update_pose()
                        
                        current_cam_idx = viewpoint_cam.cam_idx[0]
                        idx_in_list = self.all_kf_poses_idxs.index(current_cam_idx)
                        self.all_kf_poses[idx_in_list] = viewpoint_cam.c2w.numpy()
                        
                        pose_optimizer.zero_grad()
                    
                    if self.speedup:
                        if self.train_iter < self.encoder_warmup_iter or not self.is_encoder_period():
                            # Warming up OR Decoder periods: only decoder
                            self.cnn_decoder.optimizer.step()
                            self.cnn_decoder.optimizer.zero_grad(set_to_none = True)
                        else:
                            # Encoder periods: 1000-1100, 1500-1600, 2000-2100: Encoder only
                            if self.encoder_flag == 1 and self.cnn_encoder.optimizer is not None:
                                self.cnn_encoder.optimizer.step()
                                self.cnn_encoder.optimizer.zero_grad(set_to_none = True)
                    
                    if new_keyframe and self.rerun_viewer:
                        current_i = copy.deepcopy(self.iter_shared[0])
                        rgb_np = image.cpu().numpy().transpose(1,2,0)
                        rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                        rr.log("rendered_rgb", rr.Image(rgb_np))
                        new_keyframe = False


                self.training = False
                self.train_iter += 1

        if self.verbose:
            while True:
                self.run_viewer(False)
        
        # End of data
        if self.save_results and not self.rerun_viewer:
            self.gaussians.save_ply(os.path.join(self.output_path, "scene.ply"))
            if self.speedup:
                    torch.save({
                        'model_state_dict': self.cnn_decoder.state_dict(),
                        'input_dim': 16,
                        'output_dim': self.semantic_feature_dim,
                    }, os.path.join(self.output_path, "cnn_decoder.pth"))
                    if self.encoder_flag == 1:
                        torch.save({
                            'model_state_dict': self.cnn_encoder.state_dict(),
                            'input_dim': self.semantic_feature_dim,
                            'output_dim': 16,
                        }, os.path.join(self.output_path, "cnn_encoder.pth"))

        # evaluate trajectory (only if GT poses exist)
        if self.camera_parameters[8] != "femto":
            self.eval_ate()
        else:
            print("\n" + "="*50)
            print("Femto dataset: Skipping ATE evaluation (no GT poses available)")
            print("="*50)
        
        
        print(f"\n{'='*50}")
        print(f"MAPPING COMPLETED")
        print(f"Training iterations: {self.train_iter}")
        print(f"Total keyframes: {len(self.mapping_cams)}")
        print(f"Total Gaussians: {self.gaussians._xyz.shape[0]:,}")
        print(f"{'='*50}")
        

        self.calc_2d_metric()

    def rendering_loss_refinement(self):
        pass

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def pose_matrix_from_quaternion(self, pvec):
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def eval_ate(self):
        def loadReplicaPose():
            gt_file = os.path.join(self.dataset_path, 'traj.txt')
            
            pose_quat = []
            timestamps = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                quat = np.zeros(7)
                quat[:3] = c2w[:3, 3]
                quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                timestamps.append(float(i))
                pose_quat.append(quat)
                
            return np.array(pose_quat), np.array(timestamps)
        
        def loadScannetPose():
            gt_file = os.path.join(self.dataset_path, 'trajectory.txt')
            pose_quat = []
            timestamps = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    #print(line)
                    timestamps.append(float(line[0]))
                    c2w = np.array(list(map(float, line[1:]))).reshape(4, 4)
                    #print(c2w)
                    quat = np.zeros(7)
                    quat[:3] = c2w[:3, 3]
                    quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    pose_quat.append(quat)
            pose_quat = np.array(pose_quat)
            timestamps = np.array(timestamps)
            
            return pose_quat, timestamps
        
        def loadTUMPose():
            frame_rate = 32

            if os.path.isfile(os.path.join(self.dataset_path, 'groundtruth.txt')):
                pose_list = os.path.join(self.dataset_path, 'groundtruth.txt')
            elif os.path.isfile(os.path.join(self.dataset_path, 'pose.txt')):
                pose_list = os.path.join(self.dataset_path, 'pose.txt')

            image_list = os.path.join(self.dataset_path, 'rgb.txt')
            depth_list = os.path.join(self.dataset_path, 'depth.txt')

            image_data = self.parse_list(image_list)
            depth_data = self.parse_list(depth_list)
            pose_data = self.parse_list(pose_list, skiprows=1)
            pose_vecs = pose_data[:, 1:].astype(np.float64)

            tstamp_image = image_data[:, 0].astype(np.float64)
            tstamp_depth = depth_data[:, 0].astype(np.float64)
            tstamp_pose = pose_data[:, 0].astype(np.float64)
            associations = self.associate_frames(
                tstamp_image, tstamp_depth, tstamp_pose)

            indicies = [0]
            for i in range(1, len(associations)):
                t0 = tstamp_image[associations[indicies[-1]][0]]
                t1 = tstamp_image[associations[i][0]]
                if t1 - t0 > 1.0 / frame_rate:
                    indicies += [i]

            pose_quat = []
            timestamps = []
            for idx, ix in enumerate(indicies):
                (i, j, k) = associations[ix]

                c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
                
                quat = np.zeros(7)
                quat[:3] = c2w[:3, 3]
                quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                pose_quat.append(quat)
                timestamps.append(float(idx))
            
            pose_quat = np.array(pose_quat)
            timestamps = np.array(timestamps)
            
            return pose_quat, timestamps
        
        # est
        est_quats = []
        est_tstamps = []
        for idx, est_c2w in enumerate(self.all_kf_poses):
            est_quat = np.zeros(7)
            est_quat[:3] = est_c2w[:3,3]
            est_quat[3:] = Rotation.from_matrix(est_c2w[:3, :3]).as_quat()
            est_quats.append(est_quat)
            est_tstamps.append(self.all_kf_poses_idxs[idx])
        est_quats = np.array(est_quats)
        est_tstamps = np.array(est_tstamps)
        
        est_traj_evo = PoseTrajectory3D(positions_xyz=est_quats[:,:3],         
                                        orientations_quat_wxyz=est_quats[:,3:],         
                                        timestamps=est_tstamps)

        # gt
        which_dataset = self.camera_parameters[8]
        if which_dataset == "scannet":
            pose_quat, timestamps = loadScannetPose()
        elif which_dataset == "tum":
            pose_quat, timestamps = loadTUMPose()
        elif which_dataset == "replica":
            pose_quat, timestamps = loadReplicaPose()
        
        traj_ref = PoseTrajectory3D(positions_xyz=pose_quat[:,:3],         
                                        orientations_quat_wxyz=pose_quat[:,3:],         
                                        timestamps=timestamps)
        
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, est_traj_evo, max_diff=0.1)

        result = main_ape.ape(traj_ref, traj_est, est_name='traj',
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)

        # Store ATE statistics for metrics.txt
        stats = result.stats
        self.ate_rmse = stats["rmse"]
        self.ate_mean = stats["mean"]
        self.ate_max = stats["max"]
        self.ate_min = stats["min"]
        self.ate_median = stats["median"]
        self.ate_sse = stats["sse"]
        self.ate_std = stats["std"]

        print(result.pretty_str())
        # greppable one-liner for run drivers (cm, matching paper units)
        print(f"ATE_RMSE_CM: {self.ate_rmse*100:.4f}", flush=True)
        # per-frame APE (cm) vs keyframe frame index — diagnose where error spikes
        try:
            errs = result.np_arrays["error_array"]
            ts = list(traj_est.timestamps)
            order = sorted(range(len(ts)), key=lambda i: ts[i])
            for i in order:
                print(f"APE_FRAME {int(ts[i])} {errs[i]*100:.3f}", flush=True)
        except Exception as _e:
            print(f"per-frame APE skip: {_e}", flush=True)

    def update_gs_mapper(self, refined_kf_poses):
        '''
        propagate pgo update result to slam system
        1. correct poses of the living keyframe in the gs mapper (living kf)
        2. correct mean/rotations of the Gaussians (all kf)
        '''
        
        for idx, refined_pose in enumerate(refined_kf_poses):
            original_pose = self.all_kf_poses[idx]  # c2w
            
            # delta_T = np.linalg.inv(refined_pose) @ original_pose # rasterizer form
            delta_T = refined_pose @ np.linalg.inv(original_pose)

            # correct mean/rotations of the Gaussians (all kf)
            self.gaussians.transform_gaussians(idx, delta_T)

        # update kf poses
        self.all_kf_poses = refined_kf_poses
        
        for idx in range(len(self.mapping_cams)):
            cam_idx = self.mapping_cams[idx].cam_idx[0]
            cam_idx_in_list = self.all_kf_poses_idxs.index(cam_idx)
            
            T_ = torch.from_numpy(self.all_kf_poses[cam_idx_in_list])
            T = T_.inverse()
            R = T[:3,:3]
            t = T[:3,3]
            
            self.mapping_cams[idx].update_RT(R, t)
            self.mapping_cams[idx].update_matrix()

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def downsample_feature(self, semantic_feature_img, zero_filter):
        semantic_feature_point = semantic_feature_img.reshape(-1,self.semantic_feature_dim)[self.downsample_idxs]
        semantic_feature_point = semantic_feature_point[zero_filter]
        return semantic_feature_point.squeeze()

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre

    def get_image_dirs(self, images_folder):
        color_paths = []
        depth_paths = []
        semantic_features_paths =[]
        if self.trajmanager.which_dataset == "replica" or self.trajmanager.which_dataset == "femto":
            images_folder = os.path.join(images_folder, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            # Use png for femto, jpg for replica
            img_ext = ".png" if self.trajmanager.which_dataset == "femto" else ".jpg"
            # Use depth/ for femto, depth_images/ for replica
            depth_folder = "depth" if self.trajmanager.which_dataset == "femto" else "depth_images"
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                color_paths.append(f"{self.dataset_path}/images/{image_name}{img_ext}")
                depth_paths.append(f"{self.dataset_path}/{depth_folder}/{depth_image_name}.png")
                semantic_features_paths.append(f"{self.dataset_path}/rgb_feature_langseg/{image_name}_fmap_CxHxW.pt")
                
            return color_paths, depth_paths, semantic_features_paths
        elif self.trajmanager.which_dataset == "tum":
            # Generate semantic feature paths for TUM dataset
            for color_path in self.trajmanager.color_paths:
                image_filename = os.path.basename(color_path)
                image_name = os.path.splitext(image_filename)[0]  # Remove extension
                semantic_feature_name = f"{image_name}_fmap_CxHxW.pt"
                semantic_features_paths.append(f"{self.dataset_path}/rgb_feature_langseg/{semantic_feature_name}")
            
            return self.trajmanager.color_paths, self.trajmanager.depth_paths, semantic_features_paths
        elif self.trajmanager.which_dataset == "scannet":
            color_folder = os.path.join(self.dataset_path, "color")
            depth_folder = os.path.join(self.dataset_path, "depth")
            
            # Get sorted file lists by numerical order
            color_files = sorted([f for f in os.listdir(color_folder) if f.endswith('.jpg')], 
                                key=lambda x: int(os.path.splitext(x)[0]))
            depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')], 
                                key=lambda x: int(os.path.splitext(x)[0]))
            
            for i in range(min(len(color_files), len(depth_files))):
                color_file = color_files[i]
                depth_file = depth_files[i]
                
                color_path = os.path.join(color_folder, color_file)
                depth_path = os.path.join(depth_folder, depth_file)
                
                # Generate semantic feature path for ScanNet dataset
                image_name = os.path.splitext(color_file)[0]  # Remove extension -> 000000
                semantic_feature_path = f"{self.dataset_path}/rgb_feature_langseg/{image_name}_fmap_CxHxW.pt"
                
                color_paths.append(color_path)
                depth_paths.append(depth_path)
                semantic_features_paths.append(semantic_feature_path)
            
            return color_paths, depth_paths, semantic_features_paths

    def calc_2d_metric(self):
        psnrs = []
        ssims = []
        lpips = []
        accuracy_accum = []
        iou_accum = []
        
        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")

        image_names, depth_image_names, semantic_features_names = self.get_image_dirs(self.dataset_path)
        final_poses = self.final_pose

        # Apply eval_ratio to keyframes only (optimized poses after loop closing)
        total_keyframes = len(self.keyframe_idxs)
        eval_keyframes = int(total_keyframes * self.eval_ratio)
        if eval_keyframes == 0:
            eval_keyframes = 1  # At least evaluate 1 keyframe

        print(f"Evaluating {eval_keyframes}/{total_keyframes} keyframes (ratio: {self.eval_ratio})")

        # Sample keyframes evenly
        if eval_keyframes < total_keyframes:
            step = total_keyframes // eval_keyframes
            eval_indices = [self.keyframe_idxs[i] for i in range(0, total_keyframes, step)][:eval_keyframes]
        else:
            eval_indices = self.keyframe_idxs
        
        # Create directories only if image saving is enabled
        if self.save_image_flag[0] == 1:
            render_path_undistorted = os.path.join(self.output_path, "renders_undistorted")
            feature_path = os.path.join(self.output_path, "feature_map")
            gt_feature_path = os.path.join(self.output_path, "gt_feature_map")
            os.makedirs(render_path_undistorted, exist_ok=True)
            os.makedirs(feature_path, exist_ok=True)
            os.makedirs(gt_feature_path, exist_ok=True)
        else:
            render_path_undistorted = None
            feature_path = None
            gt_feature_path = None

        # ONLINE (SED+HR 768): semantic eval against the SED teacher
        # (same protocol as the offline LSeg eval below, with a 768-D teacher).
        if os.environ.get("LEGO_FEATURE_MODE", "online") == "online":
            from clip_sed.extractor import extract as sed_extract
            from clip_sed.openclip_encoder import OpenCLIPNetwork
            ade_labels = []
            with open('Lseg/label_files/ade20k_objectInfo150.txt') as lf:
                for line in lf.readlines():
                    ade_labels.append(line.strip().split(',')[-1].split(';')[0])
            ade_labels = ade_labels[1:]  # drop header (same list as offline get_labels('ade20k'))
            oc_net = OpenCLIPNetwork("cuda")
            with torch.no_grad():
                sed_text_feature = oc_net.encode_text(ade_labels, "cuda").float()  # (150, 768)
                sed_text_feature = sed_text_feature / torch.clamp(sed_text_feature.norm(dim=-1, keepdim=True), min=1e-8)

            def _sed_cls(fmap):  # (768,H,W) -> (1,H,W) argmax class map, same matching as offline
                fs = fmap.shape
                v = fmap.permute(1, 2, 0).reshape(-1, fs[0])
                v = v / torch.clamp(v.norm(dim=-1, keepdim=True).to(torch.float32), min=1e-8)
                logits = v.float() @ sed_text_feature.t()
                return logits.argmax(-1).reshape(fs[1], fs[2]).cpu().numpy()[None]

            cam = self.mapping_cams[0]
            o_psnrs, o_ssims, o_lpips = [], [], []
            o_accs, o_ious = [], []
            for i, c2w in tqdm(enumerate(self.all_kf_poses), desc="online eval"):
                kf_idx = self.all_kf_poses_idxs[i]
                # --- GT rgb/depth (identical pipeline to the offline path for a fair comparison) ---
                gt_rgb = cv2.imread(image_names[kf_idx])
                gt_depth = cv2.imread(depth_image_names[kf_idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
                if self.trajmanager.which_dataset == "scannet":
                    dh, dw = gt_depth.shape
                    gt_rgb = cv2.resize(gt_rgb, (dw, dh))
                gt_rgb = cv2.remap(gt_rgb, self.map1x, self.map1y, cv2.INTER_LINEAR, borderValue=[0, 0, 0])
                gt_mask = np.ones_like(gt_rgb, dtype=np.float32)
                gt_mask = cv2.remap(gt_mask, self.map1x, self.map1y, cv2.INTER_LINEAR, borderValue=[0, 0, 0])
                gt_mask_torch = torch.from_numpy(gt_mask).float().cuda().permute(2, 0, 1)
                gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR) / 255
                gt_rgb_ = torch.from_numpy(gt_rgb).float().cuda().permute(2, 0, 1)
                gt_depth_ = torch.from_numpy(gt_depth).float().cuda().unsqueeze(0)
                # --- render ---
                w2c = np.linalg.inv(c2w)
                cam.R = torch.tensor(w2c[:3, :3])
                cam.t = torch.tensor(w2c[:3, 3])
                cam.update_matrix()
                render_pkg = render_3(cam, self.gaussians, self.pipe, self.background)
                ours_rgb_ = torch.clamp(render_pkg["render"], 0., 1.).cuda()
                total_valid_mask = gt_mask_torch * (gt_depth_ > 0)
                gt_m = gt_rgb_ * total_valid_mask
                ours_m = ours_rgb_ * total_valid_mask
                mse_error = torch.mean(torch.mean((gt_m - ours_m) ** 2, axis=2))
                o_psnrs.append(mse2psnr(mse_error).detach().cpu())
                _, ssim_error = ssim(ours_m, gt_m)
                o_ssims.append(ssim_error.detach().cpu())
                o_lpips.append(cal_lpips(gt_m.unsqueeze(0), ours_m.unsqueeze(0)).detach().cpu())
                # --- rendered 768-D feature map ---
                fm = torch.nan_to_num(render_pkg["semantic_feature"], nan=0.0, posinf=0.0, neginf=0.0)
                fm = F.interpolate(fm.unsqueeze(0), size=(192, 192), mode='bilinear', align_corners=True).squeeze(0)
                if self.speedup:
                    fm = self.cnn_decoder(fm)  # 16 -> 768
                # --- SED-fidelity Acc/IoU (same protocol as offline LSeg eval: teacher argmax vs pred argmax) ---
                with torch.no_grad():
                    sed_in = (gt_rgb * 255).astype(np.uint8)  # RGB uint8, same as what the tracker feeds into the extractor
                    gt_fm = sed_extract(np.ascontiguousarray(sed_in)).float().cuda()  # (768,192,192)
                    gt_fm = torch.nan_to_num(gt_fm, nan=0.0, posinf=0.0, neginf=0.0)
                    pred_predict = _sed_cls(fm.detach().float().cuda())
                    gt_predict = _sed_cls(gt_fm)
                for element in (pred_predict, gt_predict):  # same label merge as offline
                    element[element == 90] = 15   # TV -> door
                    element[element == 29] = 4    # rug -> floor
                    element[element == 58] = 40   # pillow -> cushion
                m = F.interpolate(gt_mask_torch[0].unsqueeze(0).unsqueeze(0), size=(192, 192),
                                  mode='nearest').squeeze().bool().cpu().numpy()[None]
                o_accs.append(calculate_accuracy(gt_predict[m], pred_predict[m]))
                o_ious.append(calculate_iou(gt_predict[m], pred_predict[m], 7))
                del gt_fm
                torch.cuda.empty_cache()
            o_psnrs = np.array(o_psnrs); o_ssims = np.array(o_ssims); o_lpips = np.array(o_lpips)
            print(f"PSNR: {o_psnrs.mean():.2f}\nSSIM: {o_ssims.mean():.3f}\nLPIPS: {o_lpips.mean():.3f}")
            print(f"accuracy: {np.mean(o_accs):.3f}\nIOU: {np.mean(o_ious):.3f}")
            return

        ## semantic eval
        module = LSegModule.load_from_checkpoint(
            checkpoint_path="./Lseg/demo_e200.ckpt",
            data_path="./",
            dataset="ignore",
            backbone="clip_vitl16_384",
            aux="False",
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=-1,
            dropout=0.0,
            scale_inv=False,
            augment=False,
            no_batchnorm=False,
            widehead=True,
            widehead_hr=False,
            map_locatin="cpu",
            arch_option=0,
            strict=True,
            block_depth=0,
            activation="lrelu",
        )
        labels = module.get_labels('ade20k')
        num_classes = len(labels)
        input_transform = module.val_transform

        if isinstance(module.net, BaseNet):
            model = module.net
        else:
            model = module

        model = model.eval()
        model = model.cpu()

        text = clip.tokenize(labels)
        hooks = {
                "clip_vitl16_384": [5, 11, 17, 23],
                "clipRN50x16_vitl16_384": [5, 11, 17, 23],
                "clip_vitb32_384": [2, 5, 8, 11],
            }
        clip_pretrained, pretrained = make_encoder(
            "clip_vitl16_384",
            features=256,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks["clip_vitl16_384"],
            use_readout="project",
            )
        
        text = text.cuda()
        text_feature = clip_pretrained.encode_text(text)
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        with torch.no_grad():
            # for i in tqdm(eval_indices):
            for i, c2w in tqdm(enumerate(self.all_kf_poses)):
                
                gt_depth_ = []
                cam = self.mapping_cams[0]
                # c2w = final_poses[i]
                kf_idx = self.all_kf_poses_idxs[i]
                
                gt_rgb = cv2.imread(image_names[kf_idx])
                gt_depth = cv2.imread(depth_image_names[kf_idx] ,cv2.IMREAD_UNCHANGED).astype(np.float32)

                # Resize RGB image to match depth image resolution for ScanNet (like nice-slam)
                if self.trajmanager.which_dataset == "scannet":
                    depth_height, depth_width = gt_depth.shape
                    gt_rgb = cv2.resize(gt_rgb, (depth_width, depth_height))

                # undistort
                gt_rgb = cv2.remap(gt_rgb, self.map1x, self.map1y, cv2.INTER_LINEAR, borderValue=[0,0,0])

                gt_mask = np.ones_like(gt_rgb, dtype=np.float32)
                gt_mask = cv2.remap(gt_mask, self.map1x, self.map1y, cv2.INTER_LINEAR, borderValue=[0,0,0])
                gt_mask_torch = torch.from_numpy(gt_mask).float().cuda().permute(2,0,1)

                gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                gt_rgb = gt_rgb/255
                gt_rgb_ = torch.from_numpy(gt_rgb).float().cuda().permute(2,0,1)
                
                gt_depth_ = torch.from_numpy(gt_depth).float().cuda().unsqueeze(0)
                

                w2c = np.linalg.inv(c2w)
                # rendered
                R = w2c[:3,:3]
                T = w2c[:3,3]
                
                cam.R = torch.tensor(R)
                cam.t = torch.tensor(T)

                cam.update_matrix()
                render_pkg = render_3(cam, self.gaussians, self.pipe, self.background)

                ours_rgb_ = render_pkg["render"]
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()

                # Save unmasked version for saving (before mask application)
                ours_rgb_unmasked = ours_rgb_.clone()
                gt_rgb_unmasked = gt_rgb_.clone()

                valid_depth_mask_ = (gt_depth_>0)
                total_valid_mask = gt_mask_torch * valid_depth_mask_

                gt_rgb_ = gt_rgb_ * total_valid_mask
                ours_rgb_ = ours_rgb_ * total_valid_mask
                
                square_error = (gt_rgb_-ours_rgb_)**2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)
                
                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(gt_rgb_.unsqueeze(0), ours_rgb_.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]
                
                gt_feature_map = load_semantic_feature(semantic_features_names[kf_idx]).cuda()
                gt_feature_map = torch.nan_to_num(gt_feature_map, nan=0.0, posinf=0.0, neginf=0.0)

                feature_map = render_pkg["semantic_feature"]
                feature_map = torch.nan_to_num(feature_map, nan=0.0, posinf=0.0, neginf=0.0)
                feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
                if self.speedup:
                    feature_map = self.cnn_decoder(feature_map)

                feashape = feature_map.shape # (512, 360, 480)
                image_feature = feature_map.permute(1, 2, 0).reshape(-1, feashape[0]) # (172800, 512)
                # Safe normalization: add epsilon to avoid division by zero
                image_feature_norm = image_feature.norm(dim=-1, keepdim=True).to(torch.float32)
                image_feature = image_feature / torch.clamp(image_feature_norm, min=1e-8)
                text_feature = text_feature / torch.clamp(text_feature.norm(dim=-1, keepdim=True).to(torch.float32), min=1e-8)


                text_feature = text_feature.to(image_feature.device) # (150, 512)
                logit_scale = logit_scale.to(image_feature.device)
                logits_per_image = image_feature @ text_feature.t() # torch.Size([172800, 150]) logit_scale
                logits_per_image = logits_per_image.view(feashape[1], feashape[2], -1).permute(2, 0, 1)
                pred_image_text_feature = logits_per_image[None] # torch.Size([1, 150, 360, 480])

                feashape = gt_feature_map.shape # (512, 360, 480)
                image_feature = gt_feature_map.permute(1, 2, 0).reshape(-1, feashape[0]) # (172800, 512)
                # Safe normalization: add epsilon to avoid division by zero
                image_feature_norm = image_feature.norm(dim=-1, keepdim=True).to(torch.float32)
                image_feature = image_feature / torch.clamp(image_feature_norm, min=1e-8)
                text_feature = text_feature / torch.clamp(text_feature.norm(dim=-1, keepdim=True).to(torch.float32), min=1e-8)
                text_feature = text_feature.to(image_feature.device) # (150, 512)
                logit_scale = logit_scale.to(image_feature.device)
                logits_per_image = image_feature @ text_feature.t() # torch.Size([172800, 150]) logit_scale
                logits_per_image = logits_per_image.view(feashape[1], feashape[2], -1).permute(2, 0, 1)
                gt_image_text_feature = logits_per_image[None] # torch.Size([1, 150, 360, 480])
                
                pred_predict = torch.max(pred_image_text_feature, 1)[1].cpu().numpy()
                gt_predict = torch.max(gt_image_text_feature, 1)[1].cpu().numpy()

                pred_mask = utils.get_mask_pallete(pred_predict - 1, 'detail')
                gt_mask = utils.get_mask_pallete(gt_predict - 1, 'detail')
                # Visualize accumulated predictions
                pred_mask = torch.tensor(np.array(pred_mask.convert("RGB"), "f")) / 255.0
                gt_mask = torch.tensor(np.array(gt_mask.convert("RGB"), "f")) / 255.0
                gt_mask = gt_mask.cpu().detach().numpy()
                pred_mask = pred_mask.cpu().detach().numpy()

                for j in range(pred_predict.shape[1]):
                    for k in range(pred_predict.shape[2]):
                        for element in (pred_predict, gt_predict):
                            # bed sofa cushion pillow = bed
                            if element[0][j][k] == 90:  #TV to door
                                element[0][j][k] = 15
                            if element[0][j][k] == 29:  #rug to floor
                                element[0][j][k] = 4
                            if element[0][j][k] == 58:  #pillow to cushion
                                element[0][j][k] = 40

                resized_mask = F.interpolate(gt_mask_torch[0].unsqueeze(0).unsqueeze(0), 
                             size=(119, 159), 
                             mode='nearest'
                            ).squeeze().bool()
                resized_mask_np = resized_mask.cpu().numpy()
                resized_mask_np = np.expand_dims(resized_mask_np, axis=0)

                gt_predict = torch.from_numpy(gt_predict).float()
                gt_predict = F.interpolate(gt_predict.unsqueeze(0), size=(119, 159), mode='nearest').squeeze(0)
                gt_predict = gt_predict.long().numpy()  # Convert back to numpy array

                pred_predict = torch.from_numpy(pred_predict).float()
                pred_predict = F.interpolate(pred_predict.unsqueeze(0), size=(119, 159), mode='nearest').squeeze(0)
                pred_predict = pred_predict.long().numpy()

                valid_gt_pixels = gt_predict[resized_mask_np]
                valid_pred_pixels = pred_predict[resized_mask_np]

                accuracy = calculate_accuracy(valid_gt_pixels, valid_pred_pixels)
                iou = calculate_iou(valid_gt_pixels, valid_pred_pixels, 7)
                
                accuracy_accum += [accuracy]
                iou_accum += [iou]

                # Save all keyframes when save_results is enabled (each image separately)
                # Check save_image_flag: [enable, interval]
                if self.save_results and self.save_image_flag[0] == 1 and i % self.save_image_flag[1] == 0:
                    feature_map_vis = feature_visualize_saving(feature_map)
                    gt_feature_map_vis = feature_visualize_saving(gt_feature_map)

                    # Use UNMASKED version for saving (not the masked one used for evaluation)
                    ours_rgb_unmasked_np = np.asarray(ours_rgb_unmasked.detach().cpu()).squeeze().transpose((1,2,0))
                    feature_map_vis_np = np.asarray(feature_map_vis.detach().cpu()).squeeze()
                    gt_feature_map_vis_np = np.asarray(gt_feature_map_vis.detach().cpu()).squeeze()

                    # Save each image separately (not as subplot)
                    kf_idx_str = str(kf_idx).zfill(6)  # Frame index with padding (e.g., 000123)

                    # Convert rendered RGB from torch to numpy (H, W, C) in BGR format
                    ours_rgb_undistorted_bgr = cv2.cvtColor((ours_rgb_unmasked_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                    # Save rendered RGB - Undistorted version
                    cv2.imwrite(f"{render_path_undistorted}/{kf_idx_str}.png", ours_rgb_undistorted_bgr)

                    # Save GT feature visualization
                    gt_feature_bgr = cv2.cvtColor((gt_feature_map_vis_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{gt_feature_path}/{kf_idx_str}.png", gt_feature_bgr)

                    # Save rendered feature visualization
                    feature_bgr = cv2.cvtColor((feature_map_vis_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{feature_path}/{kf_idx_str}.png", feature_bgr)

                del gt_feature_map, feature_map
                torch.cuda.empty_cache()
            
            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)
            accuracy_accum = np.array(accuracy_accum)
            iou_accum = np.array(iou_accum)

            
            
            print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}\naccuracy: {accuracy_accum.mean():.3f}\nIOU: {iou_accum.mean():.3f} ")
            print(f"Evaluated on {eval_keyframes}/{total_keyframes} keyframes (ratio: {self.eval_ratio})")

            # Write comprehensive metrics to text file
            metrics_file = os.path.join(self.output_path, "metrics.txt")
            with open(metrics_file, "w") as f:
                f.write("="*60 + "\n")
                f.write("LEGO_SLAM EVALUATION RESULTS\n")
                f.write("="*60 + "\n\n")

                # Rendering metrics
                f.write("RENDERING METRICS:\n")
                f.write(f"PSNR: {psnrs.mean():.2f}\n")
                f.write(f"SSIM: {ssims.mean():.3f}\n")
                f.write(f"LPIPS: {lpips.mean():.3f}\n")
                f.write(f"Accuracy: {accuracy_accum.mean():.3f}\n")
                f.write(f"IOU: {iou_accum.mean():.3f}\n")
                f.write(f"Evaluated on {eval_keyframes}/{total_keyframes} keyframes (ratio: {self.eval_ratio})\n\n")

                # System information
                f.write("SYSTEM INFORMATION:\n")
                f.write(f"Total Keyframes: {total_keyframes}\n")
                f.write(f"Total Gaussians: {self.gaussians.get_xyz.shape[0]:,}\n")
                f.write(f"Final training iterations: {self.train_iter}\n\n")

                # Add ATE metrics if available
                if hasattr(self, 'ate_rmse') and self.ate_rmse is not None:
                    f.write("TRAJECTORY EVALUATION (ATE):\n")
                    f.write("APE w.r.t. translation part (m)\n")
                    f.write("(with SE(3) Umeyama alignment)\n\n")
                    f.write(f"     max    {self.ate_max:.6f}\n")
                    f.write(f"    mean    {self.ate_mean:.6f}\n")
                    f.write(f"  median    {self.ate_median:.6f}\n")
                    f.write(f"     min    {self.ate_min:.6f}\n")
                    f.write(f"    rmse    {self.ate_rmse:.6f}\n")
                    f.write(f"     sse    {self.ate_sse:.6f}\n")
                    f.write(f"     std    {self.ate_std:.6f}\n\n")

def make_encoder(
    backbone,
    features=256,
    use_pretrained=True,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    enable_attention_hooks=False,
):



    clip_pretrained, pretrained = _make_pretrained_clip_vitl16_384(
        use_pretrained,
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


    return clip_pretrained, pretrained

def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))


def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])

    # Replace NaN/Inf in input feature with 0
    fmap = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)

    fmap = nn.functional.normalize(fmap, dim=1)

    # Replace NaN/Inf after normalization (can occur when norm=0)
    fmap = torch.nan_to_num(fmap, nan=0.0, posinf=0.0, neginf=0.0)

    pca = sklearn.decomposition.PCA(3, random_state=42)
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()

    # Replace NaN/Inf in samples before PCA
    f_samples = np.nan_to_num(f_samples, nan=0.0, posinf=0.0, neginf=0.0)

    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1) if (q99 - q1) > 0 else 1.0  # Avoid division by zero
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div

    # Replace NaN/Inf in final result
    vis_feature = torch.nan_to_num(vis_feature, nan=0.0, posinf=0.0, neginf=0.0)

    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
    return vis_feature

def apply_distortion_to_image(img_undistorted, K, dist_coeffs):
    """Apply camera distortion to an undistorted image

    Args:
        img_undistorted: Undistorted image (H, W, C) numpy array
        K: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        img_distorted: Distorted image with same shape as input
    """
    h, w = img_undistorted.shape[:2]

    # Create mesh grid of distorted pixel coordinates
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    distorted_points = np.stack([x.ravel(), y.ravel()], axis=-1).astype(np.float32)

    # For each distorted pixel, find corresponding undistorted pixel
    # cv2.undistortPoints: distorted → undistorted
    undistorted_points = cv2.undistortPoints(
        distorted_points.reshape(-1, 1, 2),
        K,
        dist_coeffs,
        None,
        K
    )

    # Reshape to map arrays for cv2.remap
    map_x = undistorted_points[:, 0, 0].reshape(h, w).astype(np.float32)
    map_y = undistorted_points[:, 0, 1].reshape(h, w).astype(np.float32)

    # Remap: for each distorted pixel, fetch value from undistorted image
    img_distorted = cv2.remap(img_undistorted, map_x, map_y, cv2.INTER_LINEAR)

    return img_distorted

def calculate_accuracy(teacher, student):
    correct_predictions = np.sum(teacher == student)
    total_pixels = np.prod(teacher.shape)
    return correct_predictions / total_pixels

def calculate_accuracy_mask(gt, teacher, student, i):
    mask = np.equal(gt, teacher)
    mask = np.squeeze(mask)
    masked_student = np.where(mask, student, np.nan)
    correct_predictions = np.nansum(masked_student == gt)
    total_pixels = np.sum(mask)
    accuracy = correct_predictions / total_pixels
    return accuracy   
    
def calculate_iou(teacher, student, num_classes):
    iou = []

    unique_labels, counts = np.unique(np.concatenate((teacher, student)), return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]

    for i in sorted_labels[:num_classes]:
        true_labels = teacher == i
        predicted_labels = student == i
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return np.nanmean(iou)  

def calculate_iou_mask(gt, teacher, student, num_classes):
    iou = []

    unique_labels, counts = np.unique(np.concatenate((gt, teacher, student)), return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]


    matching_mask = np.equal(gt, teacher)
    for i in sorted_labels[:num_classes]:
        true_labels = (gt == i) & matching_mask
        predicted_labels = (student == i) & matching_mask
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return np.nanmean(iou) 

def load_semantic_feature(feature_path):
    """Load semantic feature from .pt file"""
    return torch.load(feature_path) 