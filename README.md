
# Bomberman RL Agent (`bomberman_rl`)

This repository contains the framework, execution scripts, and testing instructions for training and evaluating a Reinforcement Learning agent for the Bomberman competition[cite: 8].

## 🛠️ Requirements & Setup

Ensure you have Python 3 installed along with the packages specified in `requirements.txt`[cite: 8]:

```bash
pip install -r requirements.txt

```

## 🚀 Running and Testing Commands

You can run games, train your agents, or evaluate them using the `main.py` script.

### Basic Play

Watch the default rule-based agents play:

```bash
python main.py play

```

### Playing with Your Custom Agent

To test your custom agent (located in `agent_code/<your_agent>/`) against three rule-based agents:

```bash
python main.py play --my-agent <your_agent>

```

To specify multiple custom or baseline agents explicitly:

```bash
python main.py play --agents <your_agent> random_agent rule_based_agent peaceful_agent

```

### Training Your Agent

To train your agent (specifying how many leading agents are in training mode via `--train N`):

```bash
python main.py play --my-agent <your_agent> --train 1

```

You can run training faster without the graphical user interface (`--no-gui`) and using pre-configured scenarios:

```bash
python main.py play --no-gui --agents <your_agent> random_agent rule_based_agent peaceful_agent --train 1 --scenario classic

```

### Replays

To save a game replay and view it later:

```bash
python main.py play --save-replay
python main.py replay <stored-replay-file>

```

---

## ⚙️ Command-Line Arguments (`main.py play`)

* `--my-agent <str>`: Play an agent of the given name against three rule-based agents.


* `--agents <str ...>`: Explicitly set up to 4 agent names for the game.


* `--train <int>`: Sets the first $N$ agents to training mode (choices: `0`, `1`, `2`, `3`, `4`).


* `--continue-without-training`: Prevent shortening the game when training agents die.


* `--scenario <str>`: Choose the game scenario (e.g., `classic`, `coin-heaven`).


* `--seed <int>`: Set a random seed for reproducible world generation.


* `--n-rounds <int>`: Specify the number of rounds to play.


* `--save-replay`: Store the game session as a `.pt` file for playback.


* `--no-gui`: Deactivate the graphical user interface to run simulations as fast as possible.


* `--skip-frames`: Skip rendering several steps per GUI frame to speed up visual play.


* `--turn-based`: Wait for a key press before executing each movement step.



---

## 🐳 Submission Testing via Docker

To test your agent locally in the official tournament environment using Docker:

1. Build the Docker image from the directory containing the `Dockerfile`:


```bash
docker build .

```


2. Check the created image ID using `docker images`.


3. Run your agent inside the container environment:


```bash
docker run <IMAGE_ID> python main.py play --agents rule_based_agent <your_agent> --no-gui

```



```

---

### `requirements.txt`
```text
pygame
tqdm
numpy
scipy
sklearn

```
