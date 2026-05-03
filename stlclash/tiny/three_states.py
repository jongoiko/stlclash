import gymnasium as gym


def make_three_states_env() -> gym.Env:
    class StraightLineEnv(gym.Env):
        _agent_location: int

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Discrete(3, start=-1)  # Position
            self.action_space = gym.spaces.Discrete(2)  # Left/Right

        def _get_obs(self) -> int:
            return self._agent_location

        def _get_info(self) -> dict:
            return {}

        def reset(
            self, seed: int | None = None, options: dict | None = None
        ) -> tuple[int, dict]:
            super().reset()
            self._agent_location = 0
            return self._get_obs(), self._get_info()

        def step(self, action: int) -> tuple[int, int, bool, bool, dict]:
            direction = -1 if action == 0 else 1
            self._agent_location += direction
            if self._agent_location < -1:
                self._agent_location = -1
            if self._agent_location > 1:
                self._agent_location = 1
            return self._get_obs(), 0, False, False, self._get_info()

    return StraightLineEnv()
