import gymnasium as gym
import numpy as np
import ale_py

class ForceFireEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]
        self.current_lives = env.unwrapped.ale.lives()
        # NOTE: força o agente a atirar para continuar o jogo
        self.done_first_action = False

    def step(self, action: int):
        # NOTE: força o agente a atirar para continuar o jogo
        if not self.done_first_action:
            self.done_first_action = True
            return self.env.step(1)

        if self.current_lives != self.env.unwrapped.ale.lives():
            # NOTE: força o agente a atirar para continuar o jogo
            return self.env.step(1)
        else:
            return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        # NOTE: força o agente a atirar para continuar o jogo
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)

        self.done_first_action = False

        return obs, {}

