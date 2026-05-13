import gymnasium as gym


def make_straight_line_env(
    num_states: int, circular: bool, start_location: int, **kwargs: dict
) -> gym.Env:
    class StraightLineEnv(gym.Env):
        _agent_location: int
        _start_location: int
        _circular: bool
        _num_states: int

        def __init__(
            self, num_states: int, circular: bool, start_location: int, **kwargs: dict
        ) -> None:
            self.observation_space = gym.spaces.Discrete(num_states)  # Position
            self.action_space = gym.spaces.Discrete(2)  # Left/Right
            self._num_states = num_states
            self._circular = circular
            self._start_location = start_location

        def _get_obs(self) -> int:
            return self._agent_location

        def _get_info(self) -> dict:
            return {}

        def reset(
            self, seed: int | None = None, options: dict | None = None
        ) -> tuple[int, dict]:
            super().reset()
            self._agent_location = self._start_location
            return self._get_obs(), self._get_info()

        def step(self, action: int) -> tuple[int, int, bool, bool, dict]:
            direction = -1 if action == 0 else 1
            self._agent_location += direction
            if self._agent_location < 0:
                self._agent_location = 0 if not self._circular else self._num_states - 1
            if self._agent_location >= self._num_states:
                self._agent_location = 0 if self._circular else self._num_states - 1
            return self._get_obs(), 0, False, False, self._get_info()

    return StraightLineEnv(num_states, circular, start_location, **kwargs)
