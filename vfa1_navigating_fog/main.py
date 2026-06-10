from src.environment.grid_world import FogGridWorld

def run_random_episode(env: FogGridWorld, seed: int = 0) -> None:
    """Run one episode with random actions and print a per-step summary.

    Args:
        env:  An instantiated FogGridWorld.
        seed: RNG seed passed to env.reset() for reproducibility.
    """
    _, info = env.reset(seed=seed)

    print(f"{'Step':>4}  {'Pos':>8}  {'Energy':>6}  {'Radius':>6}  "
          f"{'Reward':>7}  {'Done':>5}  Action")
    print("-" * 60)

    total_reward = 0.0
    step = 0

    while True:
        action = env.action_space.sample()                 # uniform random policy
        _, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        action_name = ["UP  ", "DOWN", "LEFT", "RGHT"][action]
        done = terminated or truncated
        print(
            f"{step:>4}  {str(info['agent_pos']):>8}  "
            f"{info['energy']:>6}  {info['visible_radius']:>6}  "
            f"{reward:>+7.2f}  {str(done):>5}  {action_name}"
        )

        if done:
            break

    reason = "terminated" if terminated else "truncated (max steps)"
    print("-" * 60)
    print(f"Episode finished after {step} steps | total reward: {total_reward:.3f} | {reason}")


if __name__ == "__main__":
    env = FogGridWorld(grid_size=10)
    run_random_episode(env, seed=42)
