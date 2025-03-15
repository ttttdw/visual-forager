from gym.envs.registration import register


register(
    id='visual_foraging_gym/VisualForaging-v1.0',
    entry_point='visual_foraging_gym.envs.LargeHybridForaging:GridVisualForagingEnv',
    max_episode_steps=300,
)

register(
    id='visual_foraging_gym/VisualForaging-v1.1',
    entry_point='visual_foraging_gym.envs.LargeHybridForagingTest:GridVisualForagingEnv',
    max_episode_steps=300,
)

register(
    id='visual_foraging_gym/VisualForaging-v1.2',
    entry_point='visual_foraging_gym.envs.LargeHybridForagingUnified:GridVisualForagingEnv',
    max_episode_steps=300,
)

register(
    id='visual_foraging_gym/VisualForaging-v1.3',
    entry_point='visual_foraging_gym.envs.LargeHybridForagingValueAugmentation:GridVisualForagingEnv',
    max_episode_steps=300,
)
register(
    id='visual_foraging_gym/VisualForaging-v1.4',
    entry_point='visual_foraging_gym.envs.LargeHybridForagingFixedImage:GridVisualForagingEnv',
    max_episode_steps=300,
)
register(
    id='visual_foraging_gym/VisualForaging-v1.5',
    entry_point='visual_foraging_gym.envs.LargeHybridForagingFixedImageUnified:GridVisualForagingEnv',
    max_episode_steps=300,
)
register(
    id='visual_foraging_gym/VisualForaging-v1.6',
    entry_point='visual_foraging_gym.envs.LargeHybridForagingFixedImageTest:GridVisualForagingEnv',
    max_episode_steps=300,
)

register(
    id='visual_foraging_gym/VisualForaging-v2.0',
    entry_point='visual_foraging_gym.envs.HybridForagingTest:GridVisualForagingEnv',
    max_episode_steps=300,
)
register(
    id='visual_foraging_gym/VisualForaging-v2.1',
    entry_point='visual_foraging_gym.envs.HybridForagingTestFixImage:GridVisualForagingEnv',
    max_episode_steps=300,
)