import numpy as np

import torch
from tqdm import tqdm
import model.trainer_base as trainer_base


class AMDMTrainer(trainer_base.BaseTrainer):
    NAME = 'AMDM'

    def __init__(self, config, dataset, device):
        super(AMDMTrainer, self).__init__(config, dataset, device)
        optimizer_config = config['optimizer']
        self.full_T = optimizer_config.get('full_T', False)
        self.consistency_on = optimizer_config.get('consistency_on', False)
        self.consist_loss_weight = optimizer_config.get('consist_loss_weight', 1)
        self.loss_type = config["diffusion"]["loss_type"]
        self.recon_on = optimizer_config.get('recon_on', False)
        self.recon_loss_weight = optimizer_config.get('recon_loss_weight', 1)
        self.diffusion_loss_weight = optimizer_config.get('diffusion_loss_weight', 1)
        self.detach_step = optimizer_config.get('detach_step', 3)
        #self.foot_loss_weight = optimizer_config.get('foot_loss_weight', 0.0)

    def compute_rpr_consist_loss(self, last_frame, cur_frame):
        cur_frame_denormed = self.dataset.denorm_data(cur_frame, device=cur_frame.device)
        last_frame_denormed = self.dataset.denorm_data(last_frame, device=last_frame.device)

        jnts = self.dataset.jnts_frame_pt(cur_frame_denormed)
        if self.loss_type == 'l1':
            loss_fn = torch.nn.functional.l1_loss
        else:
            loss_fn = torch.nn.functional.mse_loss

        if 'angle' in self.dataset.data_component:
            jnts_fk = self.dataset.angle_frame_pt(cur_frame_denormed)
            consist_loss_fk_jnts = loss_fn(jnts_fk, jnts.squeeze())
        else:
            consist_loss_fk_jnts = 0

        if 'velocity' in self.dataset.data_component:
            jnts_vel = self.dataset.vel_frame_pt(last_frame_denormed, cur_frame_denormed)
            consist_loss_vel_jnts = loss_fn(jnts_vel, jnts.squeeze())
        else:
            consist_loss_vel_jnts = 0
        # consist_loss_fk_vel = loss_fn(jnts_vel, jnts_fk)
        return consist_loss_fk_jnts + consist_loss_vel_jnts

    def compute_footstep_pos_loss(self, pred_frame, gt_frame, contact_last):
        """
        pred_frame, gt_frame : [B, D] (정규화 상태, z-score)
        contact_last        : [B, 2] (L/R contact flag)

        Plan A:
          - contact_last 에서 (L or R) contact 있는 샘플만 사용
          - 그 샘플들의 toe 위치 (x,z) vs foot_cartesian (dx,dz) L2
        """
        if self.foot_loss_weight <= 0.0:
            return pred_frame.new_tensor(0.0)

        ds = self.dataset
        device = pred_frame.device

        # [B]
        contact_mask = (contact_last.sum(dim=-1) > 0).float().to(device)
        if contact_mask.sum() == 0:
            # 이번 배치에 contact 있는 프레임이 없으면 0 리턴
            return pred_frame.new_tensor(0.0)

        # 1) 정규화 해제 (denorm) : torch 상태 유지
        gt_denorm   = ds.denorm_data(gt_frame,   device=device)   # [B, D]
        pred_denorm = ds.denorm_data(pred_frame, device=device)   # [B, D]

        # 2) FK로 toe 위치 얻기 (torch 버전) : [B, J, 3]
        pred_jnts = ds.angle_frame_pt(pred_denorm)  # [B, J, 3]

        # 3) toe joint index & footstep feature index
        l_toe_idx = ds.toe_idx[0]
        r_toe_idx = ds.toe_idx[1]

        s_c, e_c = ds.foot_cartesian
        s_c, e_c = int(s_c), int(e_c)

        # 4) target: GT frame의 cartesian dx,dz (denorm 상태)
        foot_cart   = gt_denorm[..., s_c:e_c]  # [B, 4]
        left_dx_dz  = foot_cart[..., 0:2]      # [B, 2]
        right_dx_dz = foot_cart[..., 2:4]      # [B, 2]

        # 5) 예측 toe 위치 (FK 결과에서 toe joint의 x,z)
        pred_L_xz = pred_jnts[:, l_toe_idx, [0, 2]]  # [B, 2]
        pred_R_xz = pred_jnts[:, r_toe_idx, [0, 2]]  # [B, 2]

        # 6) per-sample loss 계산
        if self.loss_type == 'l1':
            per_L = torch.mean(torch.abs(pred_L_xz - left_dx_dz), dim=-1)   # [B]
            per_R = torch.mean(torch.abs(pred_R_xz - right_dx_dz), dim=-1)  # [B]
        else:
            per_L = torch.mean((pred_L_xz - left_dx_dz) ** 2, dim=-1)       # [B]
            per_R = torch.mean((pred_R_xz - right_dx_dz) ** 2, dim=-1)      # [B]

        per_sample = 0.5 * (per_L + per_R)              # [B]

        # 7) contact 있는 샘플만 평균
        foot_loss = (per_sample * contact_mask).sum() / (contact_mask.sum() + 1e-8)

        return foot_loss

    def compute_teacher_loss(self, model, sampled_frames, extra_info):
        # st_index = random.randint(0,sampled_frames.shape[1]-2)
        # print('teacher forcing')
        last_frame = sampled_frames[:, 0, :]
        ground_truth = sampled_frames[:, 1, :]

        self.optimizer.zero_grad()

        diff_loss, pred_frame = model.compute_loss(last_frame, ground_truth, None, extra_info)
        loss = self.diffusion_loss_weight * diff_loss

        #  footstep 위치 loss 추가
        #foot_loss = self.compute_footstep_pos_loss(pred_frame, ground_truth, contact_last)

        #loss = loss + self.foot_loss_weight * foot_loss

        loss.backward()
        self.optimizer.step()
        model.update()

        return {
            "diff_loss": diff_loss.item(),
            #"foot_loss": foot_loss.item()
        }

        #return {"diff_loss": diff_loss.item()}

    def compute_student_loss(self, model, sampled_frames, sch_samp_prob, extra_info):
        # print('student forcing')
        loss_diff_sum, loss_consist_sum = 0, 0

        batch_size = sampled_frames.shape[0]
        shrinked_batch_size = batch_size // model.T

        for st_index in range(self.num_rollout - 1):
            self.optimizer.zero_grad()
            next_index = st_index + 1
            ground_truth = sampled_frames[:, next_index, :]

            if self.full_T:
                shrinked_batch_size = batch_size
                if st_index == 0:
                    last_frame = sampled_frames[:, 0, :]
                    last_frame_expanded = last_frame[:, None, :].expand(-1, model.T, -1).reshape(
                        shrinked_batch_size * model.T, -1)
                else:
                    last_frame = pred_frame.detach().reshape(shrinked_batch_size, model.T, -1)[:, 0, :]
                    teacher_forcing_mask = torch.bernoulli(
                        1.0 - torch.ones(shrinked_batch_size, device=pred_frame.device) * sch_samp_prob).bool()
                    last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]
                    last_frame_expanded = last_frame[:, None, :].expand(-1, model.T, -1).reshape(
                        shrinked_batch_size * model.T, -1)

                ground_truth_expanded = ground_truth[:, None, :].expand(-1, model.T, -1).reshape(
                    shrinked_batch_size * model.T, -1)

                ts = torch.arange(0, model.T, device=self.device)
                ts = ts[None, ...].expand(shrinked_batch_size, -1).reshape(-1)

                diff_loss, pred_frame = model.compute_loss(last_frame_expanded, ground_truth_expanded, ts, extra_info)
                loss = self.diffusion_loss_weight * diff_loss

            else:
                if st_index == 0:
                    last_frame = sampled_frames[:, 0, :]
                else:
                    last_frame = pred_frame.detach()
                    teacher_forcing_mask = torch.bernoulli(
                        1.0 - torch.ones(batch_size, device=pred_frame.device) * sch_samp_prob).bool()
                    last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]
                    # last_frame_expanded = last_frame[:,None,:].expand(-1, model.T, -1).reshape(shrinked_batch_size*model.T, -1)
                    # ts = torch.zeros(batch_size, device=self.device).long()

                diff_loss_teacher, _ = model.compute_loss(sampled_frames[:, st_index, :], ground_truth, None, extra_info)
                diff_loss_student, pred_frame = model.compute_loss(last_frame, ground_truth, None, extra_info)
                diff_loss = diff_loss_student + diff_loss_teacher  # / self.num_rollout

            #foot_loss = self.compute_footstep_pos_loss(pred_frame, ground_truth, contact_last)
            #loss = self.diffusion_loss_weight * diff_loss + self.foot_loss_weight * foot_loss
            loss = self.diffusion_loss_weight * diff_loss
            loss.backward()
            self.optimizer.step()
            model.update()

            loss_diff_sum += diff_loss.item()
            #loss_foot_sum += foot_loss.item()

        return {"diff_loss": loss_diff_sum}
                #"foot_loss": loss_foot_sum}

    def train_loop(self, ep, model):
        ep_loss_dict = {}

        num_samples = 0
        self._update_lr_schedule(self.optimizer, ep - 1)

        model.train()
        pbar = tqdm(self.train_dataloader, colour='green')
        cur_samples = 1
        for frames in pbar:
            extra_info = None
            frames = frames.to(self.device).float()

            self.optimizer.zero_grad()

            if self.sample_schedule[ep] > 0:
                loss_dict = self.compute_student_loss(model, frames, self.sample_schedule[ep], extra_info=extra_info)
            else:
                loss_dict = self.compute_teacher_loss(model, frames, extra_info=extra_info)

            num_samples += cur_samples

            loss = 0
            for key in loss_dict:
                loss += loss_dict[key]
                if key not in ep_loss_dict:
                    ep_loss_dict[key] = loss_dict[key]
                else:
                    ep_loss_dict[key] += loss_dict[key]

            out_str = ' '.join(['{}:{:.4f}'.format(key, val) for key, val in loss_dict.items()])
            pbar.set_description('ep:{}, {}'.format(ep, out_str))

        for key in loss_dict:
            ep_loss_dict[key] /= num_samples

        train_info = {
            "epoch": ep,
            "sch_smp_rate": self.sample_schedule[ep],
            **ep_loss_dict
        }

        return train_info