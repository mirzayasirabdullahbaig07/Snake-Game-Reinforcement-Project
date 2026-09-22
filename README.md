# 🐍 Snake RL — Teaching an AI to Play Snake

> A self-taught Snake agent built with **Deep Q-Learning**, **PyTorch**, and **Pygame**. No hand-coded strategy — the snake starts out clueless and learns entirely through trial and error, reward by reward, game by game.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-EE4C2C?logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.6-90EE90?logo=pygame&logoColor=white)
![Status](https://img.shields.io/badge/status-actively%20training-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

<br>

## 🎬 What This Is

Watch a neural network go from bumping into walls to confidently hunting down food — all without ever being told the rules of Snake. It only knows two things: **eating feels good (+10)**, **dying feels bad (−10)** — and from that alone, it works out the rest.

This project is a hands-on demonstration of **Reinforcement Learning**, specifically **Deep Q-Learning (DQN)**, applied to a game simple enough to watch and complex enough to be genuinely interesting.

<br>

## ✨ Features

- 🧠 **Deep Q-Network** — an 11-input, 3-output neural net approximating Q-values for every possible move
- 🔁 **Experience Replay** — the agent remembers up to 100,000 past moves and learns from random batches of them, not just its most recent step
- ⚖️ **Epsilon-Greedy Exploration** — starts out making random moves to explore, gradually shifts to trusting its own judgment
- 🎯 **Reward Shaping** — small nudges for moving toward vs. away from food, so the agent learns direction-seeking behavior faster instead of wandering in circles
- ⚡ **Dynamic Difficulty** — game speed starts slow (easy to watch) and ramps up automatically as the snake's score increases
- 📈 **Live Training Plot** — a real-time matplotlib chart tracking score and rolling average across every game
- 💾 **Auto-Save** — the model checkpoints itself every time it beats its own high score
- 🖥️ **CPU-Friendly** — no GPU/CUDA required, runs on a standard laptop

<br>

## 🗂️ Project Structure

```
snake-rl/
├── game.py       # The Snake environment — the "world" the agent plays in
├── model.py      # The Q-network architecture + the training/learning step
├── agent.py      # The agent's brain: state reading, memory, decision-making — run this
├── helper.py     # Live score plotting
└── requirements.txt
```

<br>

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run training
```bash
python agent.py
```

A Pygame window opens and the snake starts playing on its own immediately. A matplotlib window tracks its score over time. Leave it running — improvement is gradual, not instant.

<br>

## 🧩 How It Actually Learns

| Step | What Happens |
|---|---|
| **1. Observe** | The agent reads 11 simple signals: danger ahead/left/right, current direction, and where the food is relative to its head |
| **2. Decide** | Early on, it picks mostly random moves (*exploration*). As it plays more games, it increasingly trusts its trained predictions (*exploitation*) |
| **3. Act** | It moves once — straight, turn right, or turn left |
| **4. Get Feedback** | `+10` for eating food, `−10` for dying, small +/− nudges for moving toward/away from food |
| **5. Learn** | The network updates immediately after every move, *and* replays a random batch of past experiences after every game — this combination is what makes DQN stable |

The core learning rule is the **Bellman equation**:

```
Q_new(s, a) = r                        if the game just ended
Q_new(s, a) = r + γ · max(Q(s'))       otherwise
```

The network is trained, over and over, to bring its predictions closer to this target.

<br>

## 🎛️ Tuning

| Want to... | Change this |
|---|---|
| Watch training faster | `BASE_SPEED`, `MAX_SPEED` in `game.py` |
| Bigger/smaller brain | `hidden_size` in `Linear_QNet(11, 256, 3)` in `agent.py` |
| Faster/slower learning | `LR`, `self.gamma` in `agent.py` |
| More/less exploration | `self.epsilon = 80 - self.n_games` in `agent.py` |
| Larger memory replay | `MAX_MEMORY`, `BATCH_SIZE` in `agent.py` |

<br>

## 📊 What to Expect

Scores are near-zero for the first ~50–80 games while the agent explores randomly — that's expected, not a bug. Improvement typically becomes visible between games 80–150, with strong runs reaching **40–60+** within a few hundred games.

<br>

## 🛣️ Roadmap Ideas

- [ ] Convolutional state representation (raw grid instead of hand-crafted features)
- [ ] Double DQN / Dueling DQN for more stable learning
- [ ] Human-playable mode alongside the AI mode
- [ ] Web-based visualization of the live Q-values

<br>

## 👤 Author

**Mirza Yasir Abdullah Baig**
*AI Engineer*

Built as a hands-on exploration of reinforcement learning — how an agent can learn purely from reward and consequence, with no explicit rules ever coded in.

<br>

## 📄 License

MIT — free to use, modify, and learn from.
