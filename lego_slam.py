import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import sys
import cv2
import numpy as np
import open3d as o3d
import time
import rerun as rr
import rerun.blueprint as rrb

# Set PYTHONPATH for multiprocessing
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Also set environment variable for child processes
os.environ['PYTHONPATH'] = current_dir + ':' + os.environ.get('PYTHONPATH', '')

from argparse import ArgumentParser
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.graphics_utils import focal2fov
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints, CNN_encoder, CNN_decoder
from gaussian_renderer import render, network_gui
from mp_Tracker import Tracker
from mp_Mapper import Mapper
from mp_Loopmanager import LoopManager

torch.multiprocessing.set_sharing_strategy('file_system')

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug


class LEGO_SLAM(SLAMParameters):
    def __init__(self, args):
        super().__init__()
        # feature mode preset: online = SED+HR live (768, no pretrained); offline = LSeg .pt (512, pretrained)
        self.feature_mode = args.feature_mode
        os.environ["LEGO_FEATURE_MODE"] = self.feature_mode
        if self.feature_mode == "online":
            args.semantic_feature_dim = 768
            args.pretrained_encoder_path = ""
            args.pretrained_decoder_path = ""
        self.dataset_path = args.dataset_path
        self.config = args.config
        self.output_path = args.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = args.verbose
        self.keyframe_th = float(args.keyframe_th)
        self.knn_max_distance = float(args.knn_maxd)
        self.overlapped_th = float(args.overlapped_th)
        self.max_correspondence_distance = float(args.max_correspondence_distance)
        self.trackable_opacity_th = float(args.trackable_opacity_th)
        self.overlapped_th2 = float(args.overlapped_th2)
        self.downsample_rate = int(args.downsample_rate)
        self.semantic_feature_dim = int(args.semantic_feature_dim)
        self.point_feature_dim = int(args.point_feature_dim)
        self.test = args.test
        self.save_results = args.save_results
        self.save_image_flag = args.save_image_flag
        self.rerun_viewer = args.rerun_viewer
        self.speedup = args.speedup
        self.semantic_feature_init = args.semantic_feature_init
        self.max_mapping_keyframes = int(args.max_mapping_keyframes)
        self.post_training_iter = int(args.post_training_iter)
        self.eval_ratio = float(args.eval_ratio)
        self.dist_threshold = float(args.dist_threshold)
        self.sim_threshold = float(args.sim_threshold)
        self.max_sample_size = int(args.max_sample_size)
        self.sample_ratio = float(args.sample_ratio)
        self.k_nearest = int(args.k_nearest)
        self.prune_min_opacity = float(args.prune_min_opacity)
        self.prune_th_cli = float(args.prune_th)
        # self.system_fps_limit = float(args.system_fps_limit) if hasattr(args, 'system_fps_limit') else 20.0
        self.system_fps_limit = float(args.system_fps_limit)
        self.pretrained_encoder_path = args.pretrained_encoder_path
        self.pretrained_decoder_path = args.pretrained_decoder_path
        self.encoder_flag = int(args.encoder_flag)
        self.encoder_warmup_iter = int(args.encoder_warmup_iter)
        self.encoder_train_interval = int(args.encoder_train_interval)
        self.encoder_train_duration = int(args.encoder_train_duration)
        self.edge_weight = float(args.edge_weight)
        self.n_trackable_keyframes = int(args.n_trackable_keyframes)
        self.pose_lr_rate = float(args.pose_lr_rate)
        self.loopclosing_global_correspondence_distance = float(args.loopclosing_global_correspondence_distance)
        self.loopclosing_local_correspondence_distance = float(args.loopclosing_local_correspondence_distance)
        self.loop_constraint_noise = float(args.loop_constraint_noise)
        self.loop_closing_start = int(args.loop_closing_start)
        
        if self.rerun_viewer:
            rr.init("3dgsviewer", spawn=True)
            # Set white background
            rr.send_blueprint(
                rrb.Spatial3DView(background=[1.0, 1.0, 1.0, 1.0])
            )
        
        camera_parameters_file = open(self.config)
        camera_parameters_ = camera_parameters_file.readlines()
        self.camera_parameters = camera_parameters_[2].split()
        self.W = int(self.camera_parameters[0])
        self.H = int(self.camera_parameters[1])
        self.fx = float(self.camera_parameters[2])
        self.fy = float(self.camera_parameters[3])
        self.cx = float(self.camera_parameters[4])
        self.cy = float(self.camera_parameters[5])
        self.depth_scale = float(self.camera_parameters[6])
        self.depth_trunc = float(self.camera_parameters[7])
        # self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)
        
        self.downsample_idxs, self.x_pre, self.y_pre, self.downsample_idxs_sparse, self.x_pre_sparse, self.y_pre_sparse = self.set_downsample_filter_multires(self.downsample_rate)
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        
        # Make test cam
        # To get memory sizes of shared_cam
        test_rgb_img, test_depth_img, test_semantic_feature_name = self.get_test_image(f"{self.dataset_path}")
        test_semantic_feature_name_tensor = torch.tensor([ord(c) for c in test_semantic_feature_name], dtype=torch.int32)
        test_points, _, _, _ = self.downsample_and_make_pointcloud(test_depth_img, test_rgb_img)

        # Get size of final poses
        num_final_poses = len(self.trajmanager.gt_poses)
        
        # Shared objects
        self.shared_cam = SharedCam(FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
                                    image=test_rgb_img, depth_image=test_depth_img, semantic_feature = test_semantic_feature_name_tensor,
                                    cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy, dataset_path=self.dataset_path)
        self.shared_new_points = SharedPoints(test_points.shape[0])
        self.shared_new_gaussians = SharedGaussians(test_points.shape[0])
        self.shared_target_gaussians = SharedTargetPoints(10000000)
        self.end_of_dataset = torch.zeros((1)).int()
        self.is_tracking_keyframe_shared = torch.zeros((1)).int()
        self.is_mapping_keyframe_shared = torch.zeros((1)).int()
        self.target_gaussians_ready = torch.zeros((1)).int()
        self.new_points_ready = torch.zeros((1)).int()
        self.final_pose = torch.zeros((num_final_poses,4,4)).float()
        self.demo = torch.zeros((1)).int()
        self.is_mapping_process_started = torch.zeros((1)).int()
        self.iter_shared = torch.zeros((1)).int()
        self.refined_pose_shared = torch.eye(4).float()
        
        if self.speedup:
            self.cnn_encoder = CNN_encoder(self.semantic_feature_dim, self.point_feature_dim, enable_training=(self.encoder_flag == 1))
            self.cnn_decoder = CNN_decoder(self.point_feature_dim, self.semantic_feature_dim)
            self.cnn_encoder.share_memory()
            self.cnn_decoder.share_memory()
        else:
            self.cnn_decoder = None
            # self.cnn_decoder_optimizer = None

        self.shared_cam.share_memory()
        self.shared_new_points.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()
        self.is_tracking_keyframe_shared.share_memory_()
        self.is_mapping_keyframe_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.final_pose.share_memory_()
        self.demo.share_memory_()
        self.is_mapping_process_started.share_memory_()
        self.iter_shared.share_memory_()
        self.refined_pose_shared.share_memory_()
        
        self.demo[0] = args.demo
        
        # Loop closing configuration from arguments
        self.enable_loop_closing = args.enable_loop_closing
        self.mapper = Mapper(self)
        self.tracker = Tracker(self)
        if self.enable_loop_closing:
            self.loop_manager = LoopManager(self)
            print("Loop Closing: ENABLED")
        else:
            self.loop_manager = None
            print("Loop Closing: DISABLED")

    def tracking(self, rank):
        self.tracker.run()
    
    def mapping(self, rank):
        self.mapper.run()
        
    def loopclosing(self, rank):
        self.loop_manager.run()

    def run(self):
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, )) 
            # elif rank == 2:
            #     p = mp.Process(target=self.loopclosing, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def get_test_image(self, images_folder):

        if self.camera_parameters[8] == "replica" or self.camera_parameters[8] == "femto":
            images_folder = os.path.join(self.dataset_path, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
            semantic_feature_name = f"{image_name}_fmap_CxHxW.pt"
            # Use png for femto, jpg for replica
            img_ext = ".png" if self.camera_parameters[8] == "femto" else ".jpg"
            # Use depth/ for femto, depth_images/ for replica
            depth_folder = "depth" if self.camera_parameters[8] == "femto" else "depth_images"
            rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}{img_ext}")
            depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/{depth_folder}/{depth_image_name}.png")).astype(np.float32)
            # semantic_feature = torch.load(f"{self.dataset_path}/rgb_feature_langseg/{semantic_feature_name}")
        elif self.camera_parameters[8] == "tum":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
            
            # Generate semantic feature name for TUM dataset
            image_name = os.path.splitext(rgb_file)[0]  # Remove extension
            semantic_feature_name = f"{image_name}_fmap_CxHxW.pt"
        elif self.camera_parameters[8] == "scannet":
            color_folder = os.path.join(self.dataset_path, "color")
            depth_folder = os.path.join(self.dataset_path, "depth")
            
            # Get sorted file lists by numerical order
            color_files = sorted([f for f in os.listdir(color_folder) if f.endswith('.jpg')], 
                                key=lambda x: int(os.path.splitext(x)[0]))
            depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')], 
                                key=lambda x: int(os.path.splitext(x)[0]))
            
            color_file = color_files[0]
            depth_file = depth_files[0]
            image_name = os.path.splitext(color_file)[0]
            
            # Load RGB and depth images
            rgb_image = cv2.imread(os.path.join(color_folder, color_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
            
            # Resize RGB image to match depth image resolution (like nice-slam)
            depth_h, depth_w = depth_image.shape
            rgb_image = cv2.resize(rgb_image, (depth_w, depth_h))
            
            semantic_feature_name = f"{image_name}_fmap_CxHxW.pt"
            # semantic_feature = torch.load(f"{self.dataset_path}/rgb_feature_langseg/{semantic_feature_name}")
        
        return rgb_image, depth_image, semantic_feature_name

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
                    
                    # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
                    # net_image = torch.concat([net_image,net_image,net_image], dim=0)
                    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

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

    def downsample_and_make_pointcloud(self, depth_img, rgb_img):
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        filter = torch.where((z_values!=0)&(z_values<=self.depth_trunc))
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values
        x = self.x_pre * z_values
        y = self.y_pre * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    
    def get_image_dirs(self, images_folder):
        if self.camera_parameters[8] == "replica" or self.camera_parameters[8] == "femto":
            images_folder = os.path.join(self.dataset_path, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
        elif self.camera_parameters[8] == "tum":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            image_files = os.listdir(rgb_folder)
            depth_files = os.listdir(depth_folder)
        elif self.camera_parameters[8] == "scannet":
            color_folder = os.path.join(self.dataset_path, "color")
            depth_folder = os.path.join(self.dataset_path, "depth")
            image_files = sorted(os.listdir(color_folder), key=lambda x: int(os.path.splitext(x)[0]))
            depth_files = sorted(os.listdir(depth_folder), key=lambda x: int(os.path.splitext(x)[0]))
 
        return image_files, depth_files


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_path / output_path / verbose")
    
    ## Basic Configuration
    parser.add_argument("--dataset_path", help="dataset path", default="dataset/Replica/room0")
    parser.add_argument("--config", help="caminfo", default="configs/Replica/caminfo.txt")
    parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save_results", action='store_true', default=None)
    parser.add_argument("--save_image_flag", type=int, nargs=2, default=[0, 2],
                        help="Image saving control: [enable, interval]. enable: 0=off, 1=on. interval: 1=all, 2=half, 3=1/3, etc.")

    ## Tracking & Mapping Parameters
    parser.add_argument("--keyframe_th", default=0.7)
    parser.add_argument("--knn_maxd", default=99999.0)
    parser.add_argument("--overlapped_th", default=5e-4)
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    
    ## Feature & Processing Parameters  
    parser.add_argument("--downsample_rate", default=10)
    parser.add_argument("--feature_mode", default="online", choices=["online", "offline"],
                        help="online: SED+HR live (768); offline: precomputed LSeg .pt (512)")
    parser.add_argument("--semantic_feature_dim", default=512)
    parser.add_argument("--point_feature_dim", default=16) 
    parser.add_argument("--semantic_feature_init", action="store_true", default=False)
    
    ## System Parameters
    # if you limited the memory, change the max_mapping_keyframes to a smaller value
    parser.add_argument("--rerun_viewer", action="store_true", default=False)
    parser.add_argument("--speedup", action="store_true", default=False)
    parser.add_argument("--system_fps_limit", default=15.0, help="system FPS limit")
    parser.add_argument("--max_mapping_keyframes", type=int, default=130, help="maximum number of keyframes to keep in mapping memory")
    
    ## Training Parameters
    parser.add_argument("--post_training_iter", default=0, help="number of post training iterations after dataset ends")
    parser.add_argument("--eval_ratio", default=0.01, help="ratio of images to use for evaluation (0.0-1.0)")
    
    ## Pruning Parameters
    # Adjust the following parameters to control the amount of pruning. 
    # Our proposed language-based pruning can remove 50% to 80% of Gaussians with only minimal loss in quality.
    parser.add_argument("--dist_threshold", default=0.05, help="distance threshold for language-based pruning")
    parser.add_argument("--sim_threshold", default=0.92, help="similarity threshold for language-based pruning")
    parser.add_argument("--max_sample_size", default=3000, type=int, help="maximum sample size for language-based pruning")
    parser.add_argument("--prune_min_opacity", default=0.005, help="min opacity threshold for geometric pruning (raise to make geo pruning aggressive)")
    parser.add_argument("--prune_th", default=-1.0, help="size pruning extent override (lower=more aggressive size pruning); -1 uses dataset default")
    parser.add_argument("--sample_ratio", default=0.1, type=float, help="sample ratio for language-based pruning")
    parser.add_argument("--k_nearest", default=100, type=int, help="k nearest neighbors for language-based pruning")
    
    ## Loop Closing Parameters
    parser.add_argument("--enable_loop_closing", action="store_true", default=True, help="enable loop closing (default: True, all scenes)")
    parser.add_argument("--disable_loop_closing", dest="enable_loop_closing", action="store_false", help="disable loop closing")
    ## Network Parameters
    parser.add_argument("--pretrained_encoder_path", help="pretrained encoder path", default="")
    parser.add_argument("--pretrained_decoder_path", help="pretrained decoder path", default="")
    parser.add_argument("--encoder_flag", type=int, default=0, help="enable encoder training (1: enable, 0: disable)")
    parser.add_argument("--encoder_warmup_iter", type=int, default=1000, help="iterations before encoder training starts")
    parser.add_argument("--encoder_train_interval", type=int, default=500, help="interval between encoder training periods")
    parser.add_argument("--encoder_train_duration", type=int, default=100, help="duration of each encoder training period")
    
    parser.add_argument("--edge_weight", help="weight of the edge matching factor", default=-1)
    parser.add_argument("--n_trackable_keyframes", default=25)
    parser.add_argument("--pose_lr_rate", default=0.5)
    parser.add_argument("--loopclosing_global_correspondence_distance", default=0.5)
    parser.add_argument("--loopclosing_local_correspondence_distance", default=0.5)
    parser.add_argument("--loop_constraint_noise", default=1e-3)
    parser.add_argument("--loop_closing_start", default=25)

    args = parser.parse_args()

    lego_slam = LEGO_SLAM(args)
    # lego_slam.SLAM(1)
    lego_slam.run()