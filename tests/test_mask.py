import pytest
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from topologies.masked_linear import MaskedLinear
from topologies.random_sparse import make_rs
from topologies.sb3_integration import create_masked_policy_kwargs, WeightClampingCallback

# @pytest.mark.skip(reason="Test is slow due to 5000 training timesteps. Remove skip to run.")
def test_masked_weights_remain_zero_after_training():
    """Test that weights in MaskedLinear layers corresponding to zeros in the mask remain zero after training."""
    # 0. Test Setup for CI speed
    torch.set_num_threads(1)

    # 1. Setup
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_name = "CartPole-v1" 
    base_env = gym.make(env_name)
    base_env.reset(seed=seed)
    env = DummyVecEnv([lambda: base_env])
    env = VecMonitor(env)

    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n
    n_hidden = 32 

    # 2. Create topology mask (random sparse to ensure some connections are masked)
    adj_mask_np, _ = make_rs(n_in, n_hidden, n_out, density=0.5, seed=seed)
    
    # Create policy_kwargs with the masked topology
    # features_dim here is the output dimension of the feature extractor (n_hidden)
    policy_kwargs = create_masked_policy_kwargs(
        adj_mask=adj_mask_np, 
        n_in=n_in,
        n_out=n_out, # n_out is used by make_rs for mask generation, not directly by MaskedFeatureExtractor
        features_dim=n_hidden, 
        hidden_act=torch.nn.ReLU(),
        out_act=torch.nn.Identity() # Not used by MaskedFeatureExtractor
    )
    policy_kwargs["net_arch"] = [] # Ensure SB3 doesn't add its own FC layers after the extractor

    # 3. Create PPO model with WeightClampingCallback
    clamping_callback = WeightClampingCallback()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        n_steps=256,       
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        seed=seed,
        verbose=0
    )

    # 4. Train for the specified number of steps
    # User requested 5k steps. This can be slow for a unit test.
    # Marking as skippable by default or using a smaller number for CI might be good.
    model.learn(total_timesteps=1000, callback=clamping_callback, progress_bar=False) # progress_bar=False for cleaner test output

    # 5. Assert masked weights are zero
    # The MaskedLinear layer is within model.policy.features_extractor (our MaskedFeatureExtractor)
    extractor = model.policy.features_extractor
    
    # grab **the first** MaskedLinear we find
    mlayer = next((m for m in extractor.modules() if isinstance(m, MaskedLinear)), None)
    assert mlayer is not None, "No MaskedLinear layer found in the feature extractor."

    # Fetch the learnable weight tensor that has the same shape as the mask
    try:
        weights_param = next(p for p in mlayer.parameters() if p.shape == mlayer.mask.shape)
        weights = weights_param.data
    except StopIteration:
        raise AssertionError("Could not find a learnable parameter with the same shape as the mask in MaskedLinear.")
    
    mask = mlayer.mask

    # Ensure the mask has both 0s and 1s for a meaningful test
    assert torch.any(mask == 0), "Mask for the layer is all ones. Test is not meaningful."
    assert torch.any(mask == 1), "Mask for the layer is all zeros. Check mask generation."

    zero_mask_elements = (mask == 0)
    masked_weights_values = weights[zero_mask_elements]
    
    assert torch.allclose(masked_weights_values, torch.zeros_like(masked_weights_values), atol=1e-7), \
        f"Masked weights in the layer are not zero. Max diff: {torch.max(torch.abs(masked_weights_values))}. Some values: {masked_weights_values[0:5]}"

    print("test_masked_weights_remain_zero_after_training passed successfully.") 