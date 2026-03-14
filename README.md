# 🚗 Virtual Self-Driving Car AI

A self-driving car simulation using Deep Q-Learning and Kivy — the agent learns to navigate a 2D map while avoiding user-drawn obstacles in real time.

![demo](https://user-images.githubusercontent.com/22739177/32823936-c279686a-c993-11e7-906e-ea3e7830e275.gif)
![demo2](https://user-images.githubusercontent.com/22739177/32823937-c2950e80-c993-11e7-9358-89e50cdaae8f.gif)

## What It Does

A small car navigates a 2D canvas, learning in real time to drive between two goal points (top-left ↔ bottom-right). Draw "sand" obstacles on the map with your mouse — the car learns to avoid them using reinforcement learning. The brain can be saved/loaded between sessions.

## Architecture

### Deep Q-Network (DQN)

| Component | Detail |
|-----------|--------|
| Network | `input(5) → fc(30, ReLU) → output(3)` |
| Replay buffer | 100,000 transitions, batch size 100 |
| Optimizer | Adam, lr = 0.001 |
| Loss | Smooth L1 (Huber) |
| γ (discount) | 0.9 |
| Exploration | Softmax with temperature T=100 |

### State Space (5 inputs)

| Input | Description |
|-------|-------------|
| `signal1` | Sand density around front sensor (20×20 px window) |
| `signal2` | Sand density around left sensor (+30°) |
| `signal3` | Sand density around right sensor (−30°) |
| `orientation` | Angle between velocity and goal direction ∈ [-1, 1] |
| `-orientation` | Negated orientation (symmetry signal) |

### Action Space

| Action | Rotation |
|--------|----------|
| 0 | 0° (straight) |
| 1 | +20° (left) |
| 2 | −20° (right) |

### Rewards

| Condition | Reward |
|-----------|--------|
| On sand | −1.0 |
| On road, moving away | −0.2 |
| On road, approaching goal | +0.1 |
| Hitting map edge | −1.0 |

Goal flips to the opposite corner when the car gets within 100 px.

## Project Structure

```
Self_Driving_Car/
├── ai.py              # DQN agent: network, replay memory, training loop
├── map_commented.py   # Kivy app: car physics, sensors, game loop, rewards
└── car.kv             # Kivy layout: car and sensor widget definitions
```

## Tech Stack

| | Technology | Purpose |
|-|------------|---------|
| 🧠 | PyTorch | Deep Q-Network |
| 🖼️ | Kivy | 2D simulation GUI |
| 🔢 | NumPy | Sensor signal processing |
| 📊 | Matplotlib | Score plotting |
| 🐍 | Python 3 | Runtime |

## Getting Started

### Install dependencies

```bash
pip install kivy torch numpy matplotlib
```

### Run

```bash
cd Self_Driving_Car
python map_commented.py
```

### Controls

- **Draw obstacles** — click and drag to paint sand
- **Clear** — remove all sand
- **Save** — save brain to `last_brain.pth` + plot scores
- **Load** — restore a previously saved brain

## Known Issues

- The simulation relies heavily on global state, making it fragile for extension but functional as a learning project.
- Sensor readings can produce index-out-of-bounds when the car is near map edges (numpy silently clips).
- No GPU support configured — runs on CPU only (sufficient for this network size).
- Button positions are hardcoded relative to initial widget size and may overlap on small windows.

## License

MIT — Kaustabh Ganguly, 2018
