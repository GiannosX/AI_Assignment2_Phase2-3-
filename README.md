This branch contains the basic structure for completing the second phase of the Second Project assigned during the AI course in TUC, during the 2025-2026 academic year. 
Two AI agents (MCTS+UCT and MCTS + informed UCT) agents should be implemented. All the agents will be tested across different domains in the [ns gym](https://nsgym.io/).

## Installation Instructions
For the installation of the project, it is suggested to use a virtual environment (such as [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)). Python 3.10 should be running in this environment.

If you have to cloned the repository, download the code of the second branch using this: 
```bash
git clone --branch MCTS --single-branch https://github.com/leoBakop/AI-agents-in-ns-gym.git 
```

It is suggested the use of an isolated environment
To install the requirements, please run:
```python
pip install -r requirements.txt
```

Also to downgrade gym (if required by the ns-gym), you should run:
```python
pip install "setuptools==65.5.0" "pip==21"
```

To test the installation, please run 
```python
python3 test/test_ns_agent.py
```

Assuming a correct installation, you should be able to see a rendering of the environment, and also the tqdm bar appeared at your terminal.