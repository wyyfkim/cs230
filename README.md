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
```

### Playing the Game with the Uploaded Model
```bash
# ppo
python3 PPO/play.py
```
- PPO saved model is in [PPO/models](https://github.com/wyyfkim/cs230/tree/main/PPO/models)

- PPO_LSTEM saved model is in [PPO_LSTM/models](https://github.com/wyyfkim/cs230/tree/main/PPO_LSTM/models)


### Notes
There is no significant performance difference between PPO and PPO_LSTM. As a result, only the PPO human play mode script has been included.

<img src="docs/ppo%20vs%20ppo_lstm.png" width="400"/>
