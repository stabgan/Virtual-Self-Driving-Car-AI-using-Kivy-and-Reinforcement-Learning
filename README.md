# Virtual Self-Driving Car AI

A self-driving car simulation built with Kivy and Deep Q-Learning (DQN), where an agent learns to navigate a 2D map while avoiding user-drawn obstacles.

## What It Does

A small car navigates a 2D canvas, learning in real time to drive between two goal points (top-left and bottom-right corners). You can draw "sand" (obstacles) on the map with your mouse, and the car learns to avoid them using reinforcement learning. The car's brain can be saved and loaded between sessions.

![demo](https://user-images.githubusercontent.com/22739177/32823936-c279686a-c993-11e7-906e-ea3e7830e275.gif)
![demo2](https://user-images.githubusercontent.com/22739177/32823937-c2950e80-c993-11e7-9358-89e50cdaae8f.gif)

## Architecture

### RL Algorithm — Deep Q-Network (DQN)

- **Network**: Simple feedforward neural net — `input(5) → fc(30, ReLU) → output(3)`
- **Experience replay**: Buffer of 100,000 transitions, sampled in batches of 100
- **Optimizer**: Adam (lr = 0.001)
- **Loss**: Smooth L1 (Huber loss)
- **Discount factor (γ)**: 0.9
- **Action selection**: Softmax over Q-values scaled by temperature T=100 (exploration via stochastic sampling, not ε-greedy)

### State Space (5 inputs)

| Input | Description |
|-------|-------------|
| `signal1` | Sand density in a 20×20 pixel window around the front sensor |
| `signal2` | Sand density around the left sensor (+30°) |
| `signal3` | Sand density around the right sensor (−30°) |
| `orientation` | Angle between car velocity and goal direction, normalized to [-1, 1] |
| `-orientation` | Negated orientation (provides symmetry signal) |

### Action Space (3 outputs)

| Action | Rotation |
|--------|----------|
| 0 | 0° (straight) |
| 1 | +20° (left) |
| 2 | −20° (right) |

### Reward Function

| Condition | Reward |
|-----------|--------|
| Driving on sand | −1.0 |
| Driving on road, moving away from goal | −0.2 |
| Driving on road, moving toward goal | +0.1 |
| Hitting any map edge | −1.0 |

When the car gets within 100 pixels of the goal, the goal flips to the opposite corner.

## Project Structure

```
Self_Driving_Car/
├── ai.py              # DQN agent: network, replay memory, learning logic
├── map_commented.py   # Kivy app: car, sensors, game loop, reward logic
└── car.kv             # Kivy layout: car and sensor widget definitions
```

## Dependencies

- Python 3.x
- [Kivy](https://kivy.org/)
- PyTorch
- NumPy
- Matplotlib

Install:

```bash
pip install kivy torch numpy matplotlib
```

## Running

```bash
cd Self_Driving_Car
python map_commented.py
```

- **Draw obstacles**: Click and drag on the canvas to paint sand
- **Clear**: Removes all sand from the map
- **Save**: Saves the trained brain to `last_brain.pth` and plots the score history
- **Load**: Restores a previously saved brain

## Known Issues and Deprecations

These are documentation-only — no code changes have been made.

1. **`Variable` with `volatile=True` is removed in modern PyTorch.** In `ai.py`, `select_action` uses `Variable(state, volatile=True)`. This was deprecated in PyTorch 0.4 and removed later. The modern equivalent is `torch.no_grad()`.

2. **`retain_variables=True` renamed to `retain_graph=True`.** In `ai.py`, `td_loss.backward(retain_variables=True)` uses the old kwarg name. Works on very old PyTorch but will fail on recent versions.

3. **`F.softmax` called without `dim` argument.** In `ai.py`, `F.softmax(...)` will raise a deprecation warning (or error in PyTorch ≥1.x) because the `dim` parameter is required.

4. **`.multinomial()` called without `num_samples`.** `probs.multinomial()` requires a `num_samples` argument in modern PyTorch.

5. **`torch.load` without `weights_only` parameter.** In recent PyTorch versions, `torch.load('last_brain.pth')` triggers a warning recommending `weights_only=True` for security.

6. **Import path mismatch.** `map_commented.py` imports `from ai import Dqn`, so the file must be named `ai.py`. This works as-is, but the original README references `ia.py` in a comment — likely a leftover from a French-named version.

7. **No `requirements.txt` or `setup.py`.** Dependencies are not formally declared.

8. **Global state throughout.** The simulation relies heavily on global variables (`sand`, `brain`, `last_reward`, etc.), which makes the code fragile but functional for a learning project.

## License

MIT — Kaustabh Ganguly, 2018
