from gymnasium.envs.registration import register

register(
    "three-states-v0",
    entry_point="stlclash.tiny.three_states:make_three_states_env",
    max_episode_steps=10,
)
