import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import os
import shutil
from env_wrapper import BusMergeEnv
from ppo_agent import PPOAgent

def evaluate_policy(agent, episodes=40, seed=123, verbose_shield=False):
    """Deterministic fixed-seed eval — always open-road scenario for comparable measurement."""
    env = BusMergeEnv(eval_mode=True)  # eval_mode: always Scenario D, never curriculum

    successes, crashes, timeouts = 0, 0, 0
    shields = 0
    shield_types = {"emergency_forward": 0, "emergency_stop": 0, "late_overlap": 0,
                    "adaptive_gap": 0, "high_pressure": 0}
    comfort_brake_steps = 0   # car decel in [-6, -4) m/s²  — yielding, natural
    emergency_brake_steps = 0 # car decel < -6 m/s²          — panic stop, undesirable
    merge_steps_sum = 0
    # adaptive_gap diagnosis: track (progress_at_trigger, gap_at_trigger, rel_speed_at_trigger)
    ag_snapshots = []

    for ep in range(episodes):
        random.seed(seed + ep)
        np.random.seed(seed + ep)
        torch.manual_seed(seed + ep)

        state = env.reset()
        done = False
        steps = 0

        while not done and steps < env.max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean, _, _ = agent.policy(state_tensor)
                # action_mean shape: [1, 2] — squeeze to numpy array [2]
                action_value = action_mean.squeeze(0).numpy()

            next_state, _, done = env.step(action_value)

            if env.last_intervened:
                shields += 1
                cond = env.last_shield_condition
                if cond in shield_types:
                    shield_types[cond] += 1
                if cond == "adaptive_gap":
                    gap, _, _, _, rel_speed, _ = env._scan_nearest_car()
                    progress = env._merge_progress()
                    ag_snapshots.append((ep, progress, gap, rel_speed, float(action_value[0])))

            _, _, car_accel, _, _, _ = env._scan_nearest_car()
            progress = float(next_state[0])  # Feature 0 = merge_progress
            if progress > 0.1:
                if car_accel < -6.0:
                    emergency_brake_steps += 1
                elif car_accel < -4.0:
                    comfort_brake_steps += 1

            state = next_state
            steps += 1

        merge_steps_sum += steps

        if env.last_success:
            successes += 1
        elif env.last_crash:
            crashes += 1
        elif env.last_timeout:
            timeouts += 1

    ep = max(1, episodes)
    if verbose_shield and ag_snapshots:
        # Sample up to 10 examples to understand when adaptive_gap fires
        sample = ag_snapshots[:10]
        print(f"  [AG-DIAGNOSIS seed={seed}] {len(ag_snapshots)} total. "
              f"Sample (ep, progress, gap_m, rel_spd, pressure):")
        for s in sample:
            print(f"    ep={s[0]:2d}  prog={s[1]:.2f}  gap={s[2]:.2f}m  rel_spd={s[3]:.2f}m/s  pressure={s[4]:.3f}")
    return {
        "successes": successes,
        "crashes": crashes,
        "timeouts": timeouts,
        "shield_per_ep": shields / ep,
        "shield_types": shield_types,
        "comfort_brake_per_ep": comfort_brake_steps / ep,
        "emergency_brake_per_ep": emergency_brake_steps / ep,
        "avg_merge_time_s": (merge_steps_sum / ep) * env.dt,
    }


def evaluate_policy_multi_seed(agent, episodes=40, seeds=(123, 456, 777), verbose_shield=False):
    runs = [evaluate_policy(agent, episodes=episodes, seed=s, verbose_shield=(verbose_shield and i == 0))
            for i, s in enumerate(seeds)]
    # Aggregate shield type counts across seeds
    agg_types = {}
    for key in runs[0]["shield_types"]:
        agg_types[key] = sum(r["shield_types"][key] for r in runs)
    return {
        "successes": sum(r["successes"] for r in runs),
        "crashes": sum(r["crashes"] for r in runs),
        "timeouts": sum(r["timeouts"] for r in runs),
        "shield_per_ep": float(np.mean([r["shield_per_ep"] for r in runs])),
        "shield_types": agg_types,
        "comfort_brake_per_ep": float(np.mean([r["comfort_brake_per_ep"] for r in runs])),
        "emergency_brake_per_ep": float(np.mean([r["emergency_brake_per_ep"] for r in runs])),
        "avg_merge_time_s": float(np.mean([r["avg_merge_time_s"] for r in runs])),
        "seed_count": len(seeds),
        "episodes_per_seed": episodes,
    }


def _load_policy_weights(agent, path):
    if not os.path.exists(path):
        return False
    try:
        state_dict = torch.load(path, weights_only=True)
    except TypeError:
        state_dict = torch.load(path)
    agent.policy.load_state_dict(state_dict)
    return True


def train(
    epochs=100,
    batch_size=2000,
    eval_episodes=40,
    eval_seeds=(123, 456, 777),
    ppo_updates=4,
    max_grad_norm=0.5,
    rollback_patience=3,
):
    env = BusMergeEnv()
    runtime_path = "rl/checkpoints/ppo_bus_brain.pth"
    best_path = "rl/checkpoints/ppo_bus_brain_best_eval.pth"

    agent = PPOAgent(filename=runtime_path)
    # Bootstrap from best-eval if available to avoid starting from a regressed last checkpoint.
    if _load_policy_weights(agent, best_path):
        print(f"Loaded best-eval policy: {best_path}")
    else:
        agent.load()

    best_eval_crashes = float('inf')
    best_eval_timeouts = float('inf')
    best_eval_success = -1
    best_eval_shields = float('inf')
    best_eval_emergency_brakes = float('inf')  # Smooth-merge quality metric
    no_improve_windows = 0

    # Initialize best-eval trackers from the ACTUAL quality of the loaded checkpoint.
    # Without this, any first eval result (even worse quality) would be saved as "best"
    # and overwrite a previously superior checkpoint.
    print("Measuring loaded checkpoint quality before training...")
    _init_stats = evaluate_policy_multi_seed(agent, episodes=eval_episodes, seeds=eval_seeds)
    best_eval_crashes = _init_stats['crashes']
    best_eval_timeouts = _init_stats['timeouts']
    best_eval_success = _init_stats['successes']
    best_eval_shields = _init_stats['shield_per_ep']
    best_eval_emergency_brakes = _init_stats['emergency_brake_per_ep']
    _tot = _init_stats['episodes_per_seed'] * _init_stats['seed_count']
    print(f"Checkpoint baseline: Succ={best_eval_success}/{_tot} Crash={best_eval_crashes} "
          f"TO={best_eval_timeouts} Shield/Ep={best_eval_shields:.2f}")
    
    print("Starting PPO PyTorch Training with Analytics...")
    
    for epoch in range(epochs):
        # --- DOMAIN RANDOMIZATION CURRICULUM ---
        # Noise sigma: start very gentle (0.05m) and grow to 0.30m by the final epoch.
        # Too much noise at Epoch 0 makes the world chaotic and prevents convergence.
        noise_sigma = 0.05 + (epoch / max(epochs - 1, 1)) * 0.25  # 0.05 → 0.30
        env.noise_sigma = noise_sigma

        # Shield penalty scale: 1.0 → 2.0 over training (D1 curriculum).
        # Early: lighter penalty = agent has "training wheels", can explore freely.
        # Late: heavier penalty = agent must find clean gaps without relying on shield.
        shield_scale = 1.0 + (epoch / max(epochs - 1, 1))  # 1.0 → 2.0
        env.shield_penalty_scale = shield_scale

        state = env.reset()
        epoch_reward = 0
        steps = 0
        
        # --- NEW: Trackers for this batch ---
        successes, crashes, timeouts, shield_blocks = 0, 0, 0, 0
        episodes_completed = 0
        
        while steps < batch_size:
            action_value, action_logprob = agent.choose_action(state)
            next_state, reward, done = env.step(action_value)
            
            # --- NEW: Track exact events based on the reward logic in env_wrapper ---
            if env.last_intervened and not done:
                shield_blocks += 1
            if done:
                episodes_completed += 1
                if env.last_success:
                    successes += 1
                elif env.last_crash:
                    crashes += 1
                elif env.last_timeout:
                    timeouts += 1
            
            agent.store_transition(state, action_value, action_logprob, reward, next_state, done)
            
            state = next_state
            epoch_reward += reward
            steps += 1
            
            if done:
                state = env.reset()
                
        # --- PHASE 2: PPO UPDATE ---
        # For 2D actions, m[1] is numpy array shape (2,); m[2] is scalar logprob.
        states       = torch.FloatTensor(np.array([m[0] for m in agent.memory], dtype=np.float32))
        actions      = torch.FloatTensor(np.array([m[1] for m in agent.memory], dtype=np.float32))  # [batch, 2]
        old_logprobs = torch.FloatTensor(np.array([m[2] for m in agent.memory], dtype=np.float32)).unsqueeze(1)  # [batch, 1]
        rewards = [m[3] for m in agent.memory]
        dones   = [m[5] for m in agent.memory]
        
        discounted_rewards = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: R = 0
            R = r + (agent.gamma * R)
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards).unsqueeze(1)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        
        with torch.no_grad():
            _, _, state_values = agent.policy(states)
            advantages = discounted_rewards - state_values

        for _ in range(ppo_updates):
            action_mean, action_std, current_state_values = agent.policy(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            # new_logprobs: [batch, 2] — sum across action dims to get per-step logprob
            new_logprobs = dist.log_prob(actions).sum(dim=-1, keepdim=True)  # [batch, 1]
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - agent.clip_ratio, 1 + agent.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_state_values, discounted_rewards)
            # Entropy bonus: prevents policy from collapsing into a single strategy
            # and keeps the agent exploring different timing windows.
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy
            
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_grad_norm)
            agent.optimizer.step()
            
        agent.memory.clear()
        
        # --- PHASE 3: NEW DETAILED LOGGING ---
        if (epoch + 1) % 10 == 0:
            avg_score = epoch_reward / episodes_completed if episodes_completed > 0 else 0
            # Print adaptive_gap diagnosis on the very first eval only
            verbose = (epoch + 1 == 10)
            eval_stats = evaluate_policy_multi_seed(agent, episodes=eval_episodes, seeds=eval_seeds, verbose_shield=verbose)
            total_eval_episodes = eval_stats["episodes_per_seed"] * eval_stats["seed_count"]
            
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            print(f"Stats  | Episodes: {episodes_completed} | Succ: {successes} | Crash: {crashes} | TO: {timeouts} | Shield: {shield_blocks}")
            print(f"Scores | Avg Score: {avg_score:.1f} | Actor Loss: {actor_loss.item():.3f} | Critic Loss: {critic_loss.item():.3f}")
            st = eval_stats['shield_types']
            print(
                "Eval  | "
                f"Succ: {eval_stats['successes']}/{total_eval_episodes} | "
                f"Crash: {eval_stats['crashes']} | TO: {eval_stats['timeouts']} | "
                f"Shield/Ep: {eval_stats['shield_per_ep']:.2f} "
                f"[ag={st['adaptive_gap']} hp={st['high_pressure']} lo={st['late_overlap']} "
                f"ef={st['emergency_forward']} es={st['emergency_stop']}] | "
                f"ComfortBrake/Ep: {eval_stats['comfort_brake_per_ep']:.2f} | "
                f"EmergencyBrake/Ep: {eval_stats['emergency_brake_per_ep']:.2f} | "
                f"MergeTime: {eval_stats['avg_merge_time_s']:.2f}s | "
                f"Seeds: {list(eval_seeds)}"
            )

            improved = False
            if eval_stats['crashes'] < best_eval_crashes:
                improved = True
            elif eval_stats['crashes'] == best_eval_crashes:
                if eval_stats['timeouts'] < best_eval_timeouts:
                    improved = True
                elif eval_stats['timeouts'] == best_eval_timeouts:
                    if eval_stats['successes'] > best_eval_success:
                        improved = True
                    # Tie-break 1: fewer shield interventions (cleaner gap selection)
                    elif eval_stats['successes'] == best_eval_success:
                        if eval_stats['shield_per_ep'] < best_eval_shields:
                            improved = True
                        # Tie-break 2: fewer emergency brakes (smoother impact on traffic)
                        elif eval_stats['shield_per_ep'] == best_eval_shields:
                            if eval_stats['emergency_brake_per_ep'] < best_eval_emergency_brakes:
                                improved = True

            if improved:
                best_eval_crashes = eval_stats['crashes']
                best_eval_timeouts = eval_stats['timeouts']
                best_eval_success = eval_stats['successes']
                best_eval_shields = eval_stats['shield_per_ep']
                best_eval_emergency_brakes = eval_stats['emergency_brake_per_ep']


                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(agent.policy.state_dict(), best_path)
                print(f"Saved new best eval model -> {best_path}")
                no_improve_windows = 0
            else:
                no_improve_windows += 1

            # Safety-first promotion: if crashes are zero on multi-seed eval, deploy this brain.
            if eval_stats['crashes'] == 0:
                shutil.copy2(best_path, runtime_path)
                print(f"Promoted zero-crash model -> {runtime_path}")

            # If evaluation regresses for several windows, roll back to best model.
            if no_improve_windows >= rollback_patience and os.path.exists(best_path):
                _load_policy_weights(agent, best_path)
                no_improve_windows = 0
                print(f"Rollback triggered: restored policy from {best_path}")

            agent.save()

    # --- FINAL VALIDATION RUN ---
    # Uses a "secret" seed set never seen during training or eval.
    # This is the true generalization test: if this score is significantly
    # lower than eval, the policy has overfit to the eval seeds.
    VALIDATION_SEEDS = (7, 31, 99, 137, 271, 314, 500, 761, 888, 1024)
    print("\n--- Final Validation Run (secret seeds, never optimised against) ---")
    _load_policy_weights(agent, best_path)
    val_stats = evaluate_policy_multi_seed(agent, episodes=eval_episodes, seeds=VALIDATION_SEEDS)
    val_total = val_stats['episodes_per_seed'] * val_stats['seed_count']
    st = val_stats['shield_types']
    print(
        f"Validation | "
        f"Succ: {val_stats['successes']}/{val_total} | "
        f"Crash: {val_stats['crashes']} | TO: {val_stats['timeouts']} | "
        f"Shield/Ep: {val_stats['shield_per_ep']:.2f} "
        f"[ag={st['adaptive_gap']} hp={st['high_pressure']} lo={st['late_overlap']} "
        f"ef={st['emergency_forward']} es={st['emergency_stop']}]"
    )
    if val_stats['crashes'] == 0 and val_stats['successes'] >= val_total * 0.98:
        print("Validation PASSED: agent generalises beyond eval seeds.")
    else:
        print("Validation WARNING: performance gap between eval and validation seeds — possible overfitting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--rollback-patience", type=int, default=10)
    parser.add_argument("--eval-seeds", type=str, default="123,456,777,42,999,2024")
    args = parser.parse_args()
    eval_seeds = tuple(int(x.strip()) for x in args.eval_seeds.split(",") if x.strip())
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        eval_seeds=eval_seeds,
        ppo_updates=args.ppo_updates,
        max_grad_norm=args.max_grad_norm,
        rollback_patience=args.rollback_patience,
    )