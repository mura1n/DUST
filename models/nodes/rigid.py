from typing import Any, Dict, List, Optional, Tuple
import logging
import random

import torch
import torch.nn as nn
from torch.nn import Parameter

from models.modules import ConditionalDeformNetwork
from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians
from utils.camera_timestamps import compute_time_offset_velocity_loss, interpolate_rigid_transforms, select_effective_group_trans

logger = logging.getLogger()

class RigidNodes(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        
    @property
    def num_instances(self):
        return self.instances_fv.shape[1]
    @property
    def num_frames(self):
        return self.instances_fv.shape[0]
    
    def get_pts_valid_mask(self):
        """
        get the mask for valid points
        """
        return self.get_instances_fv()[self.cur_frame][self.point_ids[..., 0]]

    def get_temporal_instances_fv(self):
        return self.get_instances_fv(use_original=True)
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
    def register_frame_timestamps(self, frame_timestamps: torch.Tensor):
        self.frame_timestamps = frame_timestamps.to(self.device)
    def register_pose_group_data(
        self,
        pose_group_by_cam_id: Dict[int, str],
        frame_timestamps_by_group: Dict[str, torch.Tensor],
        reference_pose_group: str,
    ):
        self.pose_group_by_cam_id = {int(k): v for k, v in pose_group_by_cam_id.items()}
        self.frame_timestamps_by_group = {
            group_name: timestamps.to(self.device)
            for group_name, timestamps in frame_timestamps_by_group.items()
        }
        self.reference_pose_group = reference_pose_group
        self.pose_groups = list(self.frame_timestamps_by_group.keys())
    def set_current_camera_context(
        self,
        frame_idx: int,
        cam_id: int,
        camera_timestamp: Optional[float] = None,
        reference_timestamp: Optional[float] = None,
    ):
        self.current_frame_idx = frame_idx
        self.current_cam_id = cam_id
        self.current_camera_timestamp = camera_timestamp
        self.current_reference_timestamp = reference_timestamp
        pose_group_by_cam_id = getattr(self, "pose_group_by_cam_id", {})
        self.current_pose_group = pose_group_by_cam_id.get(int(cam_id), getattr(self, "reference_pose_group", "shared"))

    def get_current_pose_group(self) -> str:
        if hasattr(self, "current_pose_group"):
            return self.current_pose_group
        return getattr(self, "reference_pose_group", "shared")

    def get_instances_fv(self, pose_group: Optional[str] = None, use_original: bool = False) -> torch.Tensor:
        effective_group = pose_group or self.get_current_pose_group()
        if use_original and hasattr(self, "original_instances_fv_by_group"):
            return self.original_instances_fv_by_group.get(effective_group, getattr(self, "original_instances_fv", self.instances_fv))
        if hasattr(self, "instances_fv_by_group"):
            return self.instances_fv_by_group.get(effective_group, self.instances_fv)
        return getattr(self, "original_instances_fv", self.instances_fv) if use_original else self.instances_fv

    def _interpolate_pose_at_timestamp(
        self,
        camera_timestamp: float,
        pose_group: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up instance pose at a specific real-world camera timestamp.
        Uses temporal interpolation between the two nearest known poses.
        Falls back to nearest frame if timestamp is outside the known range.
        
        For test frames (which were not optimized during training), we use
        neighboring training frames to interpolate the pose, ensuring smooth
        and reasonable poses for novel view synthesis.
        """
        group_timestamps = self.get_active_group_timestamps()
        if group_timestamps is None:
            return (
                self.get_active_instances_quats()[self.cur_frame],
                self.get_active_instances_trans()[self.cur_frame],
            )
        
        # ===== Test Frame Interpolation Fix =====
        # When rendering a test frame (eval mode), its pose at self.cur_frame was never 
        # optimized during training. We detect test frames by checking:
        # 1. We are in eval mode (not training)
        # 2. The target timestamp exactly matches the current frame's timestamp
        # 3. We have valid neighboring frames
        # In this case, we interpolate between neighboring training frames' poses.
        target_ts = float(camera_timestamp)
        cur_ts = float(group_timestamps[self.cur_frame])
        
        # Only apply test frame interpolation during evaluation (not training)
        is_eval_mode = not getattr(self, 'training', True)
        is_exact_match = abs(target_ts - cur_ts) < 1e-6
        has_valid_neighbors = (
            self.cur_frame > 0 
            and self.cur_frame < self.num_frames - 1
        )
        
        if is_eval_mode and is_exact_match and has_valid_neighbors:
            group_quats = self.get_active_instances_quats()
            group_trans = self.get_active_instances_trans()
            group_fv = self.get_instances_fv(pose_group=pose_group, use_original=True)
            
            prev_idx = self.cur_frame - 1
            next_idx = self.cur_frame + 1
            
            prev_ts = float(group_timestamps[prev_idx])
            next_ts = float(group_timestamps[next_idx])
            
            # Check if both neighbors have valid poses
            valid_mask = group_fv[prev_idx] & group_fv[next_idx]
            
            if valid_mask.any() and next_ts > prev_ts:
                # Interpolate between neighboring frames at the target timestamp position
                # The target is between prev and next, so compute the interpolation ratio
                ratio = (target_ts - prev_ts) / (next_ts - prev_ts)
                ratio_tensor = torch.tensor(
                    ratio, dtype=group_trans.dtype, device=group_trans.device
                )
                
                # Linear interpolation for translation
                interpolated_trans = (
                    group_trans[prev_idx] 
                    + (group_trans[next_idx] - group_trans[prev_idx]) * ratio_tensor
                )
                
                # Spherical interpolation for rotation
                # Clone to avoid in-place operations in interpolate_quats modifying Parameters
                interpolated_quats = interpolate_quats(
                    group_quats[prev_idx].clone(), 
                    group_quats[next_idx].clone(), 
                    fraction=ratio,
                )
                
                # Use interpolated pose where valid, fallback to current frame otherwise
                quats_cur = group_quats[self.cur_frame].clone()
                trans_cur = group_trans[self.cur_frame].clone()
                
                # Expand valid_mask to match respective shapes
                valid_mask_quats = valid_mask[:, None].expand_as(quats_cur)
                valid_mask_trans = valid_mask[:, None].expand_as(trans_cur)
                
                # Use masked operations to avoid in-place issues
                quats = torch.where(
                    valid_mask_quats, interpolated_quats, quats_cur
                )
                trans = torch.where(
                    valid_mask_trans, interpolated_trans, trans_cur
                )
                return quats, trans
        # ===== End Test Frame Interpolation Fix =====
        
        # Original logic for training frames and non-exact matches
        group_quats = self.get_active_instances_quats()
        group_trans = self.get_active_instances_trans()
        group_fv = self.get_instances_fv(pose_group=pose_group, use_original=True)
        fallback_frame = min(self.cur_frame, group_quats.shape[0] - 1)
        quats, trans = interpolate_rigid_transforms(
            instances_quats=group_quats,
            instances_trans=group_trans,
            instances_fv=group_fv,
            frame_timestamps=group_timestamps,
            fallback_frame=fallback_frame,
            target_timestamp=camera_timestamp,
        )
        return quats, trans

    def get_current_instance_poses(
        self,
        camera_timestamp: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if camera_timestamp is not None and hasattr(self, "instances_quats_by_group"):
            pose_group = self.get_current_pose_group()
            return self._interpolate_pose_at_timestamp(camera_timestamp, pose_group)
        return self.get_active_instances_quats()[self.cur_frame], self.get_active_instances_trans()[self.cur_frame]

    def get_active_instances_quats(self) -> torch.Tensor:
        if hasattr(self, "instances_quats_by_group"):
            return self.instances_quats_by_group[self.get_current_pose_group()]
        return self.instances_quats

    def get_active_instances_trans(self) -> torch.Tensor:
        if hasattr(self, "instances_trans_by_group"):
            return self.instances_trans_by_group[self.get_current_pose_group()]
        if hasattr(self, "residual_trans_by_group"):
            return select_effective_group_trans(
                instances_trans=self.instances_trans,
                residual_trans_by_group=self.residual_trans_by_group,
                pose_group=self.get_current_pose_group(),
            )
        return self.instances_trans

    def get_active_group_timestamps(self):
        if hasattr(self, "frame_timestamps_by_group"):
            return self.frame_timestamps_by_group[self.get_current_pose_group()]
        return getattr(self, "frame_timestamps", None)

    def build_initial_group_poses(
        self,
        instances_pose_by_group: Dict[str, torch.Tensor],
        instances_fv_by_group: Dict[str, torch.Tensor],
        original_instances_fv_by_group: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.pose_groups = list(instances_pose_by_group.keys())
        if not hasattr(self, "reference_pose_group") or self.reference_pose_group not in self.pose_groups:
            self.reference_pose_group = self.pose_groups[0]

        self.instances_quats_by_group = nn.ParameterDict()
        self.residual_trans_by_group = nn.ParameterDict()
        self.instances_fv_by_group = {
            pose_group: frame_valid.to(self.device)
            for pose_group, frame_valid in instances_fv_by_group.items()
        }
        self.original_instances_fv_by_group = {
            pose_group: (
                original_instances_fv_by_group.get(pose_group, instances_fv_by_group[pose_group]).to(self.device)
                if original_instances_fv_by_group is not None
                else instances_fv_by_group[pose_group].to(self.device)
            )
            for pose_group in self.pose_groups
        }

        reference_pose_stream = instances_pose_by_group[self.reference_pose_group].to(self.device)
        reference_valid_mask = self.instances_fv_by_group[self.reference_pose_group]
        reference_quats = self.get_instances_quats(reference_pose_stream, reference_valid_mask)
        self.instances_quats = Parameter(self.quat_act(reference_quats))
        self.instances_trans = Parameter(reference_pose_stream[..., :3, 3])

        for pose_group, pose_stream in instances_pose_by_group.items():
            pose_stream = pose_stream.to(self.device)
            valid_mask = self.instances_fv_by_group[pose_group]
            quats = self.get_instances_quats(pose_stream, valid_mask)
            self.instances_quats_by_group[pose_group] = Parameter(self.quat_act(quats))
            residual_trans = pose_stream[..., :3, 3] - self.instances_trans.data
            self.residual_trans_by_group[pose_group] = Parameter(residual_trans)

        self.instances_fv = self.instances_fv_by_group[self.reference_pose_group]
        self.original_instances_fv = self.original_instances_fv_by_group[self.reference_pose_group]
        
    def create_from_pcd(self, instance_pts_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        instance_pts_dict: {
            id in dataset: {
                "class_name": str,
                "pts": torch.Tensor, (N, 3)
                "colors": torch.Tensor, (N, 3)
                "poses": torch.Tensor, (num_frame, 4, 4)
                "size": torch.Tensor, (3, )
                "frame_info": torch.Tensor, (num_frame)
                "num_pts": int,
            },
        }
        """
        # collect all instances
        init_means = []
        init_colors = []
        instances_pose = []
        instances_pose_by_group = {}
        instances_size = []
        instances_fv = []
        instances_fv_by_group = {}
        original_instances_fv = []
        original_instances_fv_by_group = {}
        point_ids = []
        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            init_means.append(v["pts"])
            init_colors.append(v["colors"])
            instances_pose.append(v["poses"].unsqueeze(1))
            instances_size.append(v["size"])
            instances_fv.append(v["frame_info"].unsqueeze(1))
            if "original_frame_info" in v:
                original_instances_fv.append(v["original_frame_info"].unsqueeze(1))
            if "poses_by_group" in v:
                for pose_group, group_pose in v["poses_by_group"].items():
                    instances_pose_by_group.setdefault(pose_group, []).append(group_pose.unsqueeze(1))
                for pose_group, group_frame_info in v["frame_info_by_group"].items():
                    instances_fv_by_group.setdefault(pose_group, []).append(group_frame_info.unsqueeze(1))
                for pose_group, group_original_frame_info in v.get("original_frame_info_by_group", {}).items():
                    original_instances_fv_by_group.setdefault(pose_group, []).append(group_original_frame_info.unsqueeze(1))
            point_ids.append(torch.full((v["num_pts"], 1), id_in_model, dtype=torch.long))
        init_means = torch.cat(init_means, dim=0).to(self.device) # (N, 3)
        init_colors = torch.cat(init_colors, dim=0).to(self.device) # (N, 3)
        instances_pose = torch.cat(instances_pose, dim=1).to(self.device) # (num_frame, num_instances, 4, 4)
        self.instances_size = torch.stack(instances_size).to(self.device) # (num_instances, 3)
        self.instances_fv = torch.cat(instances_fv, dim=1).to(self.device) # (num_frame, num_instances)
        if original_instances_fv:
            self.original_instances_fv = torch.cat(original_instances_fv, dim=1).to(self.device)
        self.point_ids = torch.cat(point_ids, dim=0).to(self.device)
        grouped_pose_streams = {
            pose_group: torch.cat(group_pose_list, dim=1).to(self.device)
            for pose_group, group_pose_list in instances_pose_by_group.items()
        }
        grouped_frame_info = {
            pose_group: torch.cat(group_frame_info_list, dim=1).to(self.device)
            for pose_group, group_frame_info_list in instances_fv_by_group.items()
        }
        grouped_original_frame_info = {
            pose_group: torch.cat(group_original_frame_info_list, dim=1).to(self.device)
            for pose_group, group_original_frame_info_list in original_instances_fv_by_group.items()
        }
        
        # initialize the means, scales, quats, and colors
        self._means = Parameter(init_means)
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)
        
        # pose refinement
        if grouped_pose_streams:
            self.build_initial_group_poses(
                instances_pose_by_group=grouped_pose_streams,
                instances_fv_by_group=grouped_frame_info,
                original_instances_fv_by_group=grouped_original_frame_info,
            )
        else:
            instances_quats = self.get_instances_quats(instances_pose, self.instances_fv)
            self.instances_quats = Parameter(self.quat_act(instances_quats))
            self.instances_trans = Parameter(instances_pose[..., :3, 3])

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        if hasattr(self, "instances_quats_by_group"):
            param_groups[self.class_prefix+"ins_rotation"] = list(self.instances_quats_by_group.values())
            if hasattr(self, "residual_trans_by_group"):
                param_groups[self.class_prefix+"ins_translation"] = [self.instances_trans] + list(self.residual_trans_by_group.values())
            else:
                param_groups[self.class_prefix+"ins_translation"] = list(self.instances_trans_by_group.values())
        else:
            param_groups[self.class_prefix+"ins_rotation"] = [self.instances_quats]
            param_groups[self.class_prefix+"ins_translation"] = [self.instances_trans]
        return param_groups
    
    def get_instances_quats(self, instances_pose: torch.Tensor, valid_instances_fv: torch.Tensor) -> torch.Tensor:
        """
        Convert the pose to quaternion for all frames and instances
        """
        num_frames = instances_pose.shape[0]
        num_instances = instances_pose.shape[1]
        quats = torch.zeros(num_frames*num_instances, 4, device=self.device)
        
        poses = instances_pose[..., :3, :3].view(-1, 3, 3)
        valid_mask = valid_instances_fv.view(-1)
        _quats = matrix_to_quaternion(poses[valid_mask])
        _quats = self.quat_act(_quats)
        
        quats[valid_mask] = _quats
        quats[~valid_mask, 0] = 1.0
        return quats.reshape(num_frames, num_instances, 4)

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_ids,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_ids,
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self.point_ids = torch.cat([self.point_ids, split_ids, dup_ids], dim=0)
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            culls = culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        # step 6, sample new ids
        new_ids = self.point_ids[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_ids

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_ids = self.point_ids[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_ids

    def get_out_of_bound_mask(self):
        """
        This function checks if the gaussians are out of instance boxes
        """
        # get the instance boxes
        per_pts_size = self.instances_size[self.point_ids[..., 0]]
        instance_pts = self._means
        
        mask = (instance_pts.abs() > per_pts_size / 2).any(dim=-1)
        return mask

    def transform_means(self, means: torch.Tensor, camera_timestamp: Optional[float] = None) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        assert means.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        quats_cur_frame, trans_cur_frame = self.get_current_instance_poses(camera_timestamp=camera_timestamp)
        rot_cur_frame = quat_to_rotmat(
            self.quat_act(quats_cur_frame)
        )                                                          # (num_instances, 3, 3)
        rot_per_pts = rot_cur_frame[self.point_ids[..., 0]]        # (num_points, 3, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]
        
        # transform the means to world space
        means = torch.bmm(
            rot_per_pts, means.unsqueeze(-1)
        ).squeeze(-1) + trans_per_pts
        return means

    def transform_quats(self, quats: torch.Tensor, camera_timestamp: Optional[float] = None) -> torch.Tensor:
        """
        transform the quats of instances to world space
        according to the pose at the current frame
        """
        assert quats.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        global_quats_cur_frame, _ = self.get_current_instance_poses(camera_timestamp=camera_timestamp)
        global_quats_per_pts = global_quats_cur_frame[self.point_ids[..., 0]]
            
        global_quats_per_pts = self.quat_act(global_quats_per_pts)
        _quats = self.quat_act(quats)
        return quat_mult(global_quats_per_pts, _quats)

    def get_gaussians(self, cam: dataclass_camera, camera_timestamp: Optional[float] = None) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        # NOTE: hack here, need to consider a gaussian filter for efficient rendering
        
        world_means = self.transform_means(self._means, camera_timestamp=camera_timestamp)
        world_quats = self.transform_quats(self._quats, camera_timestamp=camera_timestamp)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        valid_mask = self.get_pts_valid_mask()
            
        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_scales = self.get_scaling
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
        
        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
        }
        return gs_dict

    def get_instance_activated_gs_dict(self, ins_id: int) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == ins_id
        if pts_mask.sum() < 100:
            return None
        local_means = self._means[pts_mask]
        activated_opacities = torch.sigmoid(self._opacities[pts_mask])
        activated_scales = torch.exp(self._scales[pts_mask])
        activated_local_rotations = self.quat_act(self._quats[pts_mask])
        gaussian_dict = {
            "means": local_means,
            "opacities": activated_opacities,            
            "scales": activated_scales,
            "quats": activated_local_rotations,
            "sh_dcs": self._features_dc[pts_mask],
            "sh_rests": self._features_rest[pts_mask],
            "ids": self.point_ids[pts_mask],
        }
        return gaussian_dict
    
    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_reg_loss()
        scaling_reg = self.reg_cfg.get("scaling_reg", None)
        if scaling_reg is not None:
            w = scaling_reg.w
            precentile = scaling_reg.precentile
            stop_after = scaling_reg.stop_after
            start_after = scaling_reg.start_after
            
            if self.step < stop_after and self.step > start_after and w > 0:
                scale_prod = self._gs_cache["_scales"].prod(dim=-1)
                p = torch.kthvalue(scale_prod, int(scale_prod.shape[0] * precentile)).values
                # penalize the scales that are too large
                loss_dict["scaling_percentile_reg"] = torch.relu(scale_prod - p).mean() * w

        # temporal smooth regularization
        temporal_smooth_reg = self.reg_cfg.get("temporal_smooth_reg", None)
        if temporal_smooth_reg is not None:
            temporal_instances_fv = self.get_temporal_instances_fv()
            instance_mask = temporal_instances_fv[self.cur_frame]
            if instance_mask.sum() > 0:
                trans_cfg = temporal_smooth_reg.get("trans", None)
                if trans_cfg is not None:
                    base_trans = self.instances_trans if hasattr(self, "residual_trans_by_group") else self.get_active_instances_trans()
                    smooth_range = int(trans_cfg.get("smooth_range", getattr(trans_cfg, "smooth_range", 1)))
                    trans_weight = float(trans_cfg.get("w", getattr(trans_cfg, "w", 0.0)))
                    if smooth_range > 0 and trans_weight > 0:
                        fi_interval = random.randint(1, smooth_range)
                        if self.cur_frame >= fi_interval and self.cur_frame < self.num_frames - fi_interval:
                            valid_mask = (
                                temporal_instances_fv[self.cur_frame - fi_interval] & \
                                temporal_instances_fv[self.cur_frame + fi_interval] & \
                                temporal_instances_fv[self.cur_frame]
                            )
                            if valid_mask.sum() > 0:
                                cur_trans = base_trans[self.cur_frame]
                                pre_trans = base_trans[self.cur_frame - fi_interval].data
                                next_trans = base_trans[self.cur_frame + fi_interval].data
                                loss = (next_trans[valid_mask] + pre_trans[valid_mask] - 2 * cur_trans[valid_mask]).abs().mean()
                                loss_dict["trans_temporal_smooth"] = loss * trans_weight

        time_offset_velocity_reg = self.reg_cfg.get("time_offset_velocity_reg", None)
        time_offset_weight = 0.0
        time_offset_smooth_range = 1
        if time_offset_velocity_reg is not None:
            time_offset_weight = float(time_offset_velocity_reg.get("w", getattr(time_offset_velocity_reg, "w", 0.0)))
            time_offset_smooth_range = int(time_offset_velocity_reg.get("smooth_range", getattr(time_offset_velocity_reg, "smooth_range", 1)))
        if (
            time_offset_velocity_reg is not None
            and time_offset_weight > 0
            and getattr(self, "current_camera_timestamp", None) is not None
        ):
            group_timestamps = self.get_active_group_timestamps()
            if group_timestamps is not None:
                loss = compute_time_offset_velocity_loss(
                    instances_trans=self.get_active_instances_trans(),
                    instances_fv=self.get_temporal_instances_fv(),
                    frame_timestamps=group_timestamps,
                    current_frame=int(self.cur_frame),
                    current_camera_timestamp=float(self.current_camera_timestamp),
                    smooth_range=time_offset_smooth_range,
                )
                if loss is not None:
                    loss_dict["time_offset_velocity_reg"] = loss * time_offset_weight

        effective_trans_temporal_smooth_reg = self.reg_cfg.get("effective_trans_temporal_smooth_reg", None)
        if effective_trans_temporal_smooth_reg is not None:
            effective_weight = float(effective_trans_temporal_smooth_reg.get("w", getattr(effective_trans_temporal_smooth_reg, "w", 0.0)))
            effective_interval = int(effective_trans_temporal_smooth_reg.get("smooth_range", getattr(effective_trans_temporal_smooth_reg, "smooth_range", 1)))
            if effective_weight > 0 and self.cur_frame >= effective_interval and self.cur_frame < self.num_frames - effective_interval:
                temporal_instances_fv = self.get_temporal_instances_fv()
                valid_mask = (
                    temporal_instances_fv[self.cur_frame - effective_interval]
                    & temporal_instances_fv[self.cur_frame + effective_interval]
                    & temporal_instances_fv[self.cur_frame]
                )
                if valid_mask.sum() > 0:
                    effective_trans = self.get_active_instances_trans()
                    cur_trans = effective_trans[self.cur_frame]
                    pre_trans = effective_trans[self.cur_frame - effective_interval].data
                    next_trans = effective_trans[self.cur_frame + effective_interval].data
                    accel = next_trans[valid_mask] + pre_trans[valid_mask] - 2 * cur_trans[valid_mask]
                    loss = torch.norm(accel, dim=-1).mean()
                    loss_dict["effective_trans_temporal_smooth"] = loss * effective_weight

        filled_residual_reg = self.reg_cfg.get("filled_residual_reg", None)
        if filled_residual_reg is not None and hasattr(self, "residual_trans_by_group"):
            filled_weight = float(filled_residual_reg.get("w", getattr(filled_residual_reg, "w", 0.0)))
            if filled_weight > 0:
                effective_group = self.get_current_pose_group()
                original_fv = self.get_instances_fv(pose_group=effective_group, use_original=True)
                repaired_fv = self.get_instances_fv(pose_group=effective_group, use_original=False)
                filled_mask = repaired_fv & ~original_fv
                if filled_mask.any() and effective_group in self.residual_trans_by_group:
                    residual_trans = self.residual_trans_by_group[effective_group]
                    loss_dict["filled_residual_reg"] = torch.norm(residual_trans[filled_mask], dim=-1).mean() * filled_weight

        group_residual_reg = self.reg_cfg.get("group_residual_reg", None)
        if group_residual_reg is not None and hasattr(self, "residual_trans_by_group"):
            residual_weight = float(group_residual_reg.get("w", getattr(group_residual_reg, "w", 0.0)))
            stop_after = int(group_residual_reg.get("stop_after", getattr(group_residual_reg, "stop_after", 0)))
            if residual_weight > 0:
                effective_group = self.get_current_pose_group()
                active_mask = self.get_instances_fv(pose_group=effective_group, use_original=False)
                if active_mask.any() and effective_group in self.residual_trans_by_group:
                    decay = 1.0
                    if stop_after > 0:
                        decay = max(0.0, 1.0 - float(self.step) / float(stop_after))
                    if decay > 0:
                        residual_trans = self.residual_trans_by_group[effective_group]
                        loss_dict["group_residual_reg"] = torch.norm(residual_trans[active_mask], dim=-1).mean() * residual_weight * decay

        return loss_dict

    def state_dict(self, *args, **kwargs) -> Dict:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({
            "points_ids": self.point_ids,
            "instances_size": self.instances_size,
            "instances_fv": self.instances_fv,
        })
        if hasattr(self, "original_instances_fv"):
            state_dict["original_instances_fv"] = self.original_instances_fv
        if hasattr(self, "instances_fv_by_group"):
            state_dict["instances_fv_by_group"] = self.instances_fv_by_group
        if hasattr(self, "original_instances_fv_by_group"):
            state_dict["original_instances_fv_by_group"] = self.original_instances_fv_by_group
        if hasattr(self, "pose_groups"):
            state_dict["pose_groups"] = self.pose_groups
        if hasattr(self, "reference_pose_group"):
            state_dict["reference_pose_group"] = self.reference_pose_group
        return state_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        self.original_instances_fv = state_dict.pop("original_instances_fv", self.instances_fv)
        self.reference_pose_group = state_dict.pop("reference_pose_group", getattr(self, "reference_pose_group", "shared"))
        self.pose_groups = state_dict.pop("pose_groups", None)
        instances_fv_by_group = state_dict.pop("instances_fv_by_group", None)
        original_instances_fv_by_group = state_dict.pop("original_instances_fv_by_group", None)
        if instances_fv_by_group is not None:
            self.instances_fv_by_group = {
                pose_group: frame_mask.to(self.device)
                for pose_group, frame_mask in instances_fv_by_group.items()
            }
            source_original_masks = original_instances_fv_by_group or instances_fv_by_group
            self.original_instances_fv_by_group = {
                pose_group: frame_mask.to(self.device)
                for pose_group, frame_mask in source_original_masks.items()
            }
            self.pose_groups = self.pose_groups or list(instances_fv_by_group.keys())
            self.instances_quats_by_group = nn.ParameterDict()
            for pose_group in self.pose_groups:
                frame_mask = self.instances_fv_by_group[pose_group]
                num_frames, num_instances = frame_mask.shape
                self.instances_quats_by_group[pose_group] = Parameter(
                    torch.zeros(num_frames, num_instances, 4, device=self.device)
                )
            self.instances_fv = self.instances_fv_by_group[self.reference_pose_group]
            self.original_instances_fv = self.original_instances_fv_by_group[self.reference_pose_group]
            residual_keys = [key for key in state_dict.keys() if key.startswith("residual_trans_by_group.")]
            if residual_keys:
                reference_mask = self.instances_fv_by_group[self.reference_pose_group]
                num_frames, num_instances = reference_mask.shape
                self.instances_trans = Parameter(
                    torch.zeros(num_frames, num_instances, 3, device=self.device)
                )
                self.residual_trans_by_group = nn.ParameterDict()
                for pose_group in self.pose_groups:
                    self.residual_trans_by_group[pose_group] = Parameter(
                        torch.zeros(num_frames, num_instances, 3, device=self.device)
                    )
            else:
                self.instances_trans_by_group = nn.ParameterDict()
                for pose_group in self.pose_groups:
                    frame_mask = self.instances_fv_by_group[pose_group]
                    num_frames, num_instances = frame_mask.shape
                    self.instances_trans_by_group[pose_group] = Parameter(
                        torch.zeros(num_frames, num_instances, 3, device=self.device)
                    )
                self.instances_trans = self.instances_trans_by_group[self.reference_pose_group]
            self.instances_quats = self.instances_quats_by_group[self.reference_pose_group]
        else:
            self.instances_trans = Parameter(
                torch.zeros(self.num_frames, self.num_instances, 3, device=self.device)
            )
            self.instances_quats = Parameter(
                torch.zeros(self.num_frames, self.num_instances, 4, device=self.device)
            )
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    
    # editting functions
    def remove_instances(self, remove_id_list: List[int]) -> None:
        """
        remove instances from the model
        
        Args:
            remove_id_list: list of instance ids to be removed
        """
        for ins_ids in remove_id_list:
            mask = ~(self.point_ids[..., 0] == ins_ids)
            self._means = Parameter(self._means[mask])
            self._scales = Parameter(self._scales[mask])
            self._quats = Parameter(self._quats[mask])
            self._features_dc = Parameter(self._features_dc[mask])
            self._features_rest = Parameter(self._features_rest[mask])
            self._opacities = Parameter(self._opacities[mask])
            self.point_ids = self.point_ids[mask]
        
    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = {}
        for id in ids:
            if id not in gaussian_dict:
                instance_raw_dict = {
                    "_means": self._means[self.point_ids[..., 0] == id],
                    "_scales": self._scales[self.point_ids[..., 0] == id],
                    "_quats": self._quats[self.point_ids[..., 0] == id],
                    "_features_dc": self._features_dc[self.point_ids[..., 0] == id],
                    "_features_rest": self._features_rest[self.point_ids[..., 0] == id],
                    "_opacities": self._opacities[self.point_ids[..., 0] == id],
                    "point_ids": self.point_ids[self.point_ids[..., 0] == id],
                }
                gaussian_dict[id] = instance_raw_dict
        return gaussian_dict

    def replace_instances(self, replace_dict: Dict[int, int]) -> None:
        """
        replace instances from the model
        
        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(replace_dict.values())
        for ins_id, new_id in replace_dict.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians_dict[new_id]
            self._means = Parameter(torch.cat([self._means, new_gaussian["_means"]], dim=0))
            self._scales = Parameter(torch.cat([self._scales, new_gaussian["_scales"]], dim=0))
            self._quats = Parameter(torch.cat([self._quats, new_gaussian["_quats"]], dim=0))
            self._features_dc = Parameter(torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0))
            self._features_rest = Parameter(torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0))
            self._opacities = Parameter(torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0))
            # keeps original point ids
            self.point_ids = torch.cat([self.point_ids, torch.full_like(new_gaussian["point_ids"], ins_id)], dim=0)
    
    def export_gaussians_to_ply(self, alpha_thresh: float, instance_id: List[int] = None) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == instance_id
        
        means = self._means[pts_mask]
        direct_color = self.colors[pts_mask]
        
        activated_opacities = self.get_opacity[pts_mask]
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }
