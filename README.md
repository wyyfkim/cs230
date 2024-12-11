# cs230 Final Project

## Get started


### Prerequisites
* Python >= 3.7.1, < 3.11

### Install Dependencies
Start by installing the necessary dependencies:

```bash
# dqn dependencies
pip install -r requirements/requirements-dqn.txt

# ppo dependencies
pip install -r requirements/requirements-ppo.txt
```

### Running Training Scripts
```bash
# ppo
python3 PPO/train.py

# ppo + lstm
python3 PPO_LSTM/main.py

# DQN
python3 DeepQNetwork/main.py

# double DQN
python3 DoubleDeepQNetwork/main.py

# C51 DQN
python3 C51DeepQNetwork/main.py

# Rainbow DQN
python3 RainbowDeepQNetwork/main.py
```

### Playing the Game with the Uploaded Model
```bash
# ppo
python3 PPO/play.py
```
- PPO saved model is in [PPO/models](https://github.com/wyyfkim/cs230/tree/main/PPO/models)

- PPO_LSTEM saved model is in [PPO_LSTM/models](https://github.com/wyyfkim/cs230/tree/main/PPO_LSTM/models)

DQN saved model is in [checkpoints](https://github.com/wyyfkim/cs230/tree/main/checkpoints)

### Notes
There is no significant performance difference between PPO and PPO_LSTM. As a result, only the PPO human play mode script has been included.

<img src="docs/ppo%20vs%20ppo_lstm.png" width="400"/>
