import torch
import torch.nn as nn
import torch.optim as optim
from policy.learning.storage import RolloutStorage
from policy.common.misc_utils import update_exponential_schedule, update_linear_schedule
import util.logging as logging_util
import util.save as save_util
import copy
import yaml
from policy.common.misc_utils import EpisodeRunner



class PPOAgent(object):
    NAME = 'PPO'

    def __init__(self, config, actor_critic, env, device):
        self.mirror_function = None
        self.config = config
        self.device = device
        self.env = env
        self.actor_critic = actor_critic.to(self.device)

        self.num_parallel = env.num_parallel
        self.mini_batch_size = config["mini_batch_size"]

        num_frames = 10e9
        self.num_steps_per_rollout = self.env.max_timestep
        self.num_updates = int(num_frames / self.num_parallel / self.num_steps_per_rollout)
        self.num_mini_batch = int(self.num_parallel * self.num_steps_per_rollout / self.mini_batch_size)
        self.num_epoch = 0
        obs_shape = self.env.observation_space.shape
        obs_shape = (obs_shape[0], *obs_shape[1:])

        self.rollouts = RolloutStorage(
            self.num_steps_per_rollout,
            self.num_parallel,
            obs_shape,
            self.actor_critic.actor.action_dim,
            self.actor_critic.state_size,
        )

        self.use_gae = config["use_gae"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]

        self.clip_param = config["clip_param"]
        self.ppo_epoch = config["ppo_epoch"]
        self.value_loss_coef = config["value_loss_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.lr = config["lr"]
        self.final_lr = config["final_lr"]
        self.lr_decay_type = config["lr_decay_type"]
        self.eps = config["eps"]
        self.save_interval = config["save_interval"]

        self.action_steps = self.env.config['action_step']
        self.action_rgr_steps = [self.action_steps.index(s) + 1 for s in self.env.config.get('action_rgr_step', [])]
        self.action_mask = torch.zeros(self.mini_batch_size, len(self.action_steps) + 1, self.env.frame_dim).to(self.device)

        if len(self.action_rgr_steps) > 0:
            self.action_mask[:, self.action_rgr_steps] = 1
        self.action_mask = self.action_mask.view(self.mini_batch_size, -1)
        self.actor_reg_weight = config.get('actor_reg_weight', 1)
        self.actor_bound_weight = config.get('actor_bound_weight', 0.0)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=self.eps)
        if not self.env.is_rendered:
            self.logger = logging_util.wandbLogger(run_name=env.int_output_dir,
                                                   proj_name="HCONTROL_{}_{}_{}_{}".format(self.NAME, env.NAME,
                                                                                           env.model.NAME,
                                                                                           env.dataset.NAME, self.NAME))

        # ---- Resume training (optional) ----
        resume_path = config.get("resume_from", None)

        self.start_update = 0

        if resume_path is not None and resume_path != "":
            print(f"[PPO] Resume from {resume_path}")

            ckpt = torch.load(resume_path, map_location=self.device)

            # save_weight는 state_dict만 저장함
            if isinstance(ckpt, dict):
                self.actor_critic.load_state_dict(ckpt)
            else:
                raise RuntimeError("Unsupported checkpoint format")

            print("[PPO] Model weights loaded.")

    def test_controller(self):
        # ---- set test parallel ----
        self.num_parallel = self.env.num_parallel_test
        obs = self.env.reset()

        device = obs.device
        B = obs.shape[0]
        N = getattr(self.env, "map_N", 16)

        # ---- map stats (assume last 2*N*N dims are maps) ----
        map_dim = 2 * N * N
        if obs.shape[1] >= map_dim:
            maps = obs[:, -map_dim:]
            mw = maps[:, : N * N].reshape(B, N, N)
            my = maps[:, N * N:].reshape(B, N, N)
            print("[Map stats]")
            print("  mw min/max/mean:", float(mw.min()), float(mw.max()), float(mw.mean()))
            print("  my min/max/mean:", float(my.min()), float(my.max()), float(my.mean()))
        else:
            print(f"[WARN] obs_dim({obs.shape[1]}) < map_dim({map_dim}). Map stats skipped.")

        # ---- episode accumulators per env ----
        ep_return = torch.zeros((B, 1), device=device, dtype=torch.float32)

        # ---- counters (python ints) ----
        n_success = 0
        n_collision = 0
        n_time_limit = 0
        n_episodes = 0

        # optional: keep recent episode returns for quick mean
        recent_returns = []

        # Some envs might require this; harmless otherwise
        try:
            self.env.reset_initial_frames()
        except Exception:
            pass

        with EpisodeRunner(self.env) as runner:
            while not runner.done:
                with torch.no_grad():
                    action = self.actor_critic.actor(obs)

                obs, reward, done, info = self.env.step(action)

                # reward: (B,1) expected
                if reward.dim() == 1:
                    reward = reward.view(-1, 1)
                ep_return += reward

                # info flags
                time_limit = info.get("reset", None)
                success = info.get("success", None)
                collision = info.get("collision", None)

                if time_limit is None:
                    time_limit = torch.zeros_like(done, dtype=torch.bool)
                elif not torch.is_tensor(time_limit):
                    time_limit = torch.tensor(time_limit, device=device).view(1, 1).expand_as(done)
                else:
                    if time_limit.dim() == 1:
                        time_limit = time_limit.view(-1, 1)

                if success is None:
                    success = torch.zeros_like(done, dtype=torch.bool)
                elif not torch.is_tensor(success):
                    success = torch.tensor(success, device=device).view(1, 1).expand_as(done)
                else:
                    if success.dim() == 1:
                        success = success.view(-1, 1)

                # done reason split
                done_t = done if done.dim() == 2 else done.view(-1, 1)
                done_success = done_t & success
                done_time = done_t & time_limit
                done_collision = done_t & collision

                # ---- print per-event summaries (optional: can comment out for speed) ----
                if done_success.any():
                    vals = ep_return[done_success]
                    mean_ret = float(vals.mean()) if vals.numel() > 0 else 0.0
                    print(f"--- SUCCESS episodes: {int(done_success.sum())} | mean return: {mean_ret:.4f}")

                if done_collision.any():
                    vals = ep_return[done_collision]
                    mean_ret = float(vals.mean()) if vals.numel() > 0 else 0.0
                    print(f"--- COLLISION episodes: {int(done_collision.sum())} | mean return: {mean_ret:.4f}")

                if done_time.any():
                    vals = ep_return[done_time]
                    mean_ret = float(vals.mean()) if vals.numel() > 0 else 0.0
                    print(f"--- TIME_LIMIT episodes: {int(done_time.sum())} | mean return: {mean_ret:.4f}")

                # ---- update counters ----
                n_s = int(done_success.sum().item())
                n_c = int(done_collision.sum().item())
                n_t = int(done_time.sum().item())
                if (n_s + n_c + n_t) > 0:
                    n_success += n_s
                    n_collision += n_c
                    n_time_limit += n_t
                    n_episodes += (n_s + n_c + n_t)

                    # store recent returns (detached to cpu)
                    done_any = done_t.squeeze(1)
                    recent_returns.extend(ep_return[done_any].detach().cpu().view(-1).tolist())

                # ---- reset done envs (indices only) ----
                if done_t.any():
                    done_indices = self.env.parallel_ind_buf.masked_select(done_t.squeeze(1))
                    # reset done envs
                    obs = self.env.reset_index(done_indices)
                    # clear ep_return for those envs
                    ep_return.index_fill_(0, done_indices, 0.0)

    def compute_action_bound_loss(self, norm_a, bound_min=-1, bound_max=1):
        violation_min = torch.clamp_max(norm_a.mean() - bound_min, 0.0)
        violation_max = torch.clamp_min(norm_a.mean() - bound_max, 0)
        bound_violation_loss = torch.sum(torch.square(violation_min), dim=-1) \
                               + torch.sum(torch.square(violation_max), dim=-1)
        return bound_violation_loss.mean()

    def compute_action_reg_weight(self, norm_a, mask):
        norm_a = norm_a * mask
        action_reg_loss = torch.sum(torch.square(norm_a), dim=-1)
        return action_reg_loss.mean()

    def _to_bool_tensor(self, x, device, like):
        """info에서 reset/success/collision을 bool tensor (B,1)로 통일."""
        if x is None:
            return torch.zeros_like(like, dtype=torch.bool)
        if torch.is_tensor(x):
            t = x.to(device=device, dtype=torch.bool)
            if t.dim() == 1:
                t = t.view(-1, 1)
            return t
        # python bool/np 등
        t = torch.tensor(x, device=device, dtype=torch.bool)
        if t.numel() == 1:
            t = t.view(1, 1).expand_as(like)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        return t

    def test_controller_panda3d(self):
        """
        Panda3D viewer로 env를 진행하면서 policy로 action을 뽑아주는 테스트.
        - viewer.consume_step_permission()이 True일 때만 1 스텝(혹은 frame_skip 루프) 진행
        - done env들은 reset_index()로 즉시 리셋
        """

        # ---- set test parallel ----
        self.num_parallel = self.env.num_parallel_test
        obs = self.env.reset()

        # Some envs might require this; harmless otherwise
        try:
            self.env.reset_initial_frames()
        except Exception:
            pass

        device = obs.device
        B = obs.shape[0]
        N = getattr(self.env, "map_N", 16)

        # ---- map stats (optional) ----
        map_dim = 2 * N * N
        if obs.shape[1] >= map_dim:
            maps = obs[:, -map_dim:]
            mw = maps[:, : N * N].reshape(B, N, N)
            my = maps[:, N * N:].reshape(B, N, N)
            print("[Map stats]")
            print("  mw min/max/mean:", float(mw.min()), float(mw.max()), float(mw.mean()))
            print("  my min/max/mean:", float(my.min()), float(my.max()), float(my.mean()))
        else:
            print(f"[WARN] obs_dim({obs.shape[1]}) < map_dim({map_dim}). Map stats skipped.")

        # ---- episode accumulators per env ----
        ep_return = torch.zeros((B, 1), device=device, dtype=torch.float32)

        # ---- counters ----
        n_success = 0
        n_collision = 0
        n_time_limit = 0
        n_episodes = 0
        recent_returns = []

        # ---- Panda3D viewer ----
        from render.realtime.mocap_renderer_panda3d import MocapPandaViewer
        viewer = MocapPandaViewer(self.env)

        runner = EpisodeRunner(self.env)
        runner.__enter__()  # with 없이 직접 enter

        # task 함수에서 obs를 갱신해야 해서 nonlocal 사용
        def step_env(task):
            nonlocal obs, ep_return, n_success, n_collision, n_time_limit, n_episodes, recent_returns

            # runner가 끝났으면 종료
            if runner.done:
                runner.__exit__(None, None, None)
                return task.done

            # viewer input이 허락할 때만 step 진행
            if not viewer.consume_step_permission():
                return task.cont

            # ---- action sample ----
            with torch.no_grad():
                action = self.actor_critic.actor(obs)

            # ---- env step ----
            obs, reward, done, info = self.env.step(action)

            if reward.dim() == 1:
                reward = reward.view(-1, 1)
            ep_return += reward

            done_t = done if done.dim() == 2 else done.view(-1, 1)

            # info flags
            time_limit = self._to_bool_tensor(info.get("reset", None), device, done_t)
            success = self._to_bool_tensor(info.get("success", None), device, done_t)
            collision = self._to_bool_tensor(info.get("collision", None), device, done_t)

            done_success = done_t & success
            done_time = done_t & time_limit
            done_collision = done_t & collision

            # ---- prints ----
            if done_success.any():
                vals = ep_return[done_success]
                mean_ret = float(vals.mean()) if vals.numel() > 0 else 0.0
                print(f"--- SUCCESS episodes: {int(done_success.sum())} | mean return: {mean_ret:.4f}")

            if done_collision.any():
                vals = ep_return[done_collision]
                mean_ret = float(vals.mean()) if vals.numel() > 0 else 0.0
                print(f"--- COLLISION episodes: {int(done_collision.sum())} | mean return: {mean_ret:.4f}")

            if done_time.any():
                vals = ep_return[done_time]
                mean_ret = float(vals.mean()) if vals.numel() > 0 else 0.0
                print(f"--- TIME_LIMIT episodes: {int(done_time.sum())} | mean return: {mean_ret:.4f}")

            # ---- update counters ----
            n_s = int(done_success.sum().item())
            n_c = int(done_collision.sum().item())
            n_t = int(done_time.sum().item())
            if (n_s + n_c + n_t) > 0:
                n_success += n_s
                n_collision += n_c
                n_time_limit += n_t
                n_episodes += (n_s + n_c + n_t)

                done_any = done_t.squeeze(1)
                recent_returns.extend(ep_return[done_any].detach().cpu().view(-1).tolist())

            # ---- reset done envs ----
            if done_t.any():
                done_indices = self.env.parallel_ind_buf.masked_select(done_t.squeeze(1))
                obs = self.env.reset_index(done_indices)
                ep_return.index_fill_(0, done_indices, 0.0)

            # ---- Panda3D update ----
            viewer.update_from_env()

            # (선택) 주기적 요약
            if n_episodes > 0 and (n_episodes % 50 == 0):
                rr = recent_returns[-200:]
                mean_rr = float(np.mean(rr)) if len(rr) > 0 else 0.0
                print(
                    f"[Summary] episodes={n_episodes} | "
                    f"success={n_success} collision={n_collision} time_limit={n_time_limit} | "
                    f"recent_mean_return(last200)={mean_rr:.3f}"
                )

            return task.cont

        # Panda3D task 등록 & run
        viewer.taskMgr.add(step_env, "step_env")
        viewer.run()  # blocking

        # 끝났으면 최종 요약
        rr = recent_returns
        mean_rr = float(np.mean(rr)) if len(rr) > 0 else 0.0
        std_rr = float(np.std(rr)) if len(rr) > 0 else 0.0
        print("========== TEST SUMMARY ==========")
        print(f"episodes:    {n_episodes}")
        print(f"success:     {n_success}")
        print(f"collision:   {n_collision}")
        print(f"time_limit:  {n_time_limit}")
        if n_episodes > 0:
            print(f"success_rate:   {n_success / n_episodes:.3f}")
            print(f"collision_rate: {n_collision / n_episodes:.3f}")
            print(f"time_rate:      {n_time_limit / n_episodes:.3f}")
        print(f"return mean/std over episodes: {mean_rr:.3f} / {std_rr:.3f}")
        print("==================================")

    def train_controller(self, out_model_file, int_output_dir):
        obs = self.env.reset()
        self.rollouts.observations[0].copy_(obs)
        self.rollouts.to(self.device)
        num_samples = 0
        #for update in range(self.num_updates):
        start = getattr(self, "start_update", 0)
        for update in range(start, self.num_updates):

            ep_info = {"reward": []}
            ep_reward = 0

            if self.lr_decay_type == "linear":
                update_linear_schedule(
                    self.optimizer, update, self.num_updates, self.lr, self.final_lr
                )
            elif self.lr_decay_type == "exponential":
                update_exponential_schedule(
                    self.optimizer, update, 0.99, self.lr, self.final_lr
                )

            for step in range(self.num_steps_per_rollout):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(
                        self.rollouts.observations[step]
                    )

                obs, reward, done, info = self.env.step(action)

                ep_reward += reward

                # ---- flags ----
                time_limit = info.get("reset", None)
                if torch.is_tensor(time_limit):
                    time_limit_t = time_limit
                else:
                    time_limit_t = torch.tensor(time_limit, device=done.device).view(1, 1).expand_as(done)

                # done is already (success|collision|time_limit)
                done_t = done if done.dim() == 2 else done.view(-1, 1)

                # ---- PPO masks ----
                masks = (~done_t).float()
                true_terminal = done_t & (~time_limit_t)  # success/collision
                bad_masks = (~true_terminal).float()

                # ---- logging episodic returns----
                done_indices = None
                if done_t.any():
                    ep_info["reward"].append(ep_reward[done_t].clone())
                    ep_reward *= (~done_t).float()
                    done_indices = self.env.parallel_ind_buf.masked_select(done_t.squeeze())

                # ---- time-limit rollout reset indices ----
                rollout_indices = None
                if time_limit_t.any():
                    rollout_indices = self.env.parallel_ind_buf.masked_select(time_limit_t.squeeze())
                    ep_reward.index_fill_(0, rollout_indices, 0.0)

                # ---- reset union ----
                if (done_indices is not None) or (rollout_indices is not None):
                    if done_indices is None:
                        reset_indices = rollout_indices
                    elif rollout_indices is None:
                        reset_indices = done_indices
                    else:
                        reset_indices = torch.unique(
                            torch.cat([done_indices.view(-1), rollout_indices.view(-1)], dim=0))
                    obs = self.env.reset_index(reset_indices)

                self.rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)

            num_samples += (obs.shape[0] * self.num_steps_per_rollout)
            with torch.no_grad():
                next_value = self.actor_critic.get_value(self.rollouts.observations[-1]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda)

            value_loss, action_loss, dist_entropy, regr = self.update(self.rollouts)

            self.rollouts.after_update()

            save_util.save_weight(copy.deepcopy(self.actor_critic), out_model_file)
            if update % self.save_interval == 0:
                save_util.save_weight(copy.deepcopy(self.actor_critic), int_output_dir + '/_ep{}.pth'.format(update))

            ep_info["reward"] = torch.cat(ep_info["reward"])

            stats = {
                "update": update,
                "reward_mean": torch.mean(ep_info['reward']),
                "reward_max": torch.max(ep_info['reward']),
                "reward_min": torch.min(ep_info['reward']),
                "dist_entropy": dist_entropy,
                "value_loss": value_loss,
                "action_loss": action_loss,
                "regr": regr,
            }
            self.logger.log_epoch(stats, step=int(num_samples))
            self.logger.print_log(stats)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        std = advantages.std()
        print("[dbg] adv std:", float(std))
        advantages = (advantages - advantages.mean()) / (std + 1e-5)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        regr_last = 0.0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                if self.mirror_function is not None:
                    (observations_batch, actions_batch, return_batch, masks_batch,
                     old_action_log_probs_batch, adv_targ) = self.mirror_function(sample)
                else:
                    (observations_batch, actions_batch, return_batch, masks_batch,
                     old_action_log_probs_batch, adv_targ) = sample

                # --- forward ---
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    observations_batch, actions_batch
                )

                # --- ratio (overflow 방지) ---
                diff = action_log_probs - old_action_log_probs_batch

                diff_clamped = torch.clamp(diff, -20.0, 20.0)
                ratio = torch.exp(diff_clamped)

                # --- PPO losses ---
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                regr = 0.0
                if self.actor_reg_weight > 0.0:
                    action_rgr_loss = self.actor_reg_weight * self.compute_action_reg_weight(
                        actions_batch, self.action_mask
                    )
                    action_loss += action_rgr_loss
                    regr += float(action_rgr_loss.detach())

                if self.actor_bound_weight > 0.0:
                    action_bound_loss = self.actor_bound_weight * self.compute_action_bound_loss(actions_batch)
                    action_loss += action_bound_loss
                    regr += float(action_bound_loss.detach())

                value_loss = (return_batch - values).pow(2).mean()

                # --- step ---
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += float(value_loss.detach())
                action_loss_epoch += float(action_loss.detach())
                dist_entropy_epoch += float(dist_entropy.detach())
                regr_last = regr  # 마지막 값 리턴용

        num_updates = self.ppo_epoch * self.num_mini_batch
        return value_loss_epoch / num_updates, action_loss_epoch / num_updates, dist_entropy_epoch / num_updates, regr_last
