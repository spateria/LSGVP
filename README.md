# LSGVP: Value-Based Subgoal Discovery and Path Planning for Reaching Long-Horizon Goals.
This is the official code for the following paper published in IEEE Transactions on Neural Networks and Learning Systems [Value-Based Subgoal Discovery and Path Planning for Reaching Long-Horizon Goals](https://ieeexplore.ieee.org/abstract/document/10040536) 

This work addresses the challenge of training autonomous agents to reach long-horizon goals in spatial traversal tasks. It introduces a novel planning method called "Learning Subgoal Graph using Value-based Subgoal Discovery and Automatic Pruning" (LSGVP). Unlike existing methods, LSGVP uses a subgoal discovery heuristic based on cumulative reward, resulting in sparse subgoals that align with higher cumulative reward paths. Additionally, LSGVP includes an automatic pruning mechanism to remove erroneous connections between subgoals, particularly those across obstacles. As a result, LSGVP outperforms other methods in terms of achieving higher cumulative rewards and goal-reaching success rates in spatial traversal tasks.

# Dependencies
Python >= 3.5.0

tensorflow==2.1.0

tf-agents==0.4.0

tensorflow-probability==0.9.0

# End-to-end training and testing
In end-to-end training and testing, the following three phases are executed in the same proram run:
1. Pre-training of RL agent policy and universal Q-value function (defined in `agent.py`, `actor_critic.py`, and `train_agent.py`).
2. Subgoal graph learning and pruning using the universal Q-value function learned in phase 1 (refer to `LSGVP.py`).
3. Testing using various long-horizon goals, using the learned RL agent and subgoal graph (refer to `test_goalseeking.py`).

Running the end-to-end program is very simple, just call
`python main.py` 

The file `main.py` contains the parameters/arguments and calls to training, subgoal graph construction, and testing functions.

# Modular execution
Coming soon...independent/asynchornous execution of the three phases by saving and loading agent policy, Q function, and subgoal graph models.
