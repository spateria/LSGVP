import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import tensorflow as tf

from env.twoD_mazes import plot_walls

def run_goalseeking(env, agent, lsgvp):
    
    difficulty = 0.8 #goal difficulty...determines the distance of sampled goal from start state
    
    num_methods = 2
    success = [[] for _ in range(num_methods)]
    
    for trial in tqdm.tnrange(25, desc='trial'):
    
        env.pyenv.envs[0]._duration = 300
        seed = np.random.randint(0, 1000000)
    
        
        max_goal_dist = env.pyenv.envs[0].gym.max_goal_dist
        env.pyenv.envs[0].gym.set_sample_goal_args(
            prob_constraint=1.0,
            min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
            max_dist=max_goal_dist * (difficulty + 0.05))
    
    
        plt.figure(figsize=(12, 5))
        
        for mthd in range(num_methods):
          
          use_lsgvp = (mthd == 1)
          
          if mthd == 0:
            title = 'no subgoals'
          elif mthd == 1:
            title = 'SoRB'
          else:
            title = 'Ours'
          
          plt.subplot(1, num_methods, mthd + 1)
          plot_walls(env.pyenv.envs[0].env.walls)
    
          np.random.seed(seed)
          ts = env.reset()
          goal = ts.observation['goal'].numpy()[0]
          #start = ts.observation['observation'].numpy()[0]
    
          obs_vec = []
          lsgvp_subgoals = []
          g_path = None
          way_g = None
      
          step = 0
          for _ in range(env.pyenv.envs[0]._duration): #tqdm.tnrange(env.pyenv.envs[0]._duration,
                                #desc='rollout %d / 2' % (mthd + 1)):
            if ts.is_last():
              success[mthd].append(1)
              break
            elif step == env.pyenv.envs[0]._duration - 2:
              success[mthd].append(0)
    
            obs_vec.append(ts.observation['observation'].numpy()[0])
    
            if use_lsgvp:
    
              g_reached = lsgvp.localize(ts.observation['observation'].numpy()[0], 
                                d_limit=5, check_g=way_g)
              
              if (step % (lsgvp.subgoal_edge_cutoff+10) == 0):
    
                stt = time.time()
                g_path = lsgvp.plan(ts.observation['observation'].numpy()[0], 
                                    ts.observation['goal'].numpy()[0], min_edge=True)
                if g_path is None:
                   stt = time.time()
                   g_path = lsgvp.plan(ts.observation['observation'].numpy()[0], 
                                    ts.observation['goal'].numpy()[0], min_edge=False)
                ett=time.time()
    
              if g_path is not None:
                '''print(g_path)
                print(ett-stt)
                sys.exit()'''
    
                if g_reached == tuple(g_path[1]):
                  g_path.pop(0) ### if a subgoal is reached on the path, make it the new start subgoal
                                        #### check at each step
                  assert tuple(g_path[0]) == g_reached
                  #print(g_path)
    
                if len(g_path)==1:
                  way_g = tuple(copy.deepcopy(ts.observation['goal'].numpy()[0]))
                  g_path = None
                else:
                  way_g = g_path[1] ### 0 is the start state
    
                lsgvp_subgoals.append(way_g)
                our_ts = copy.deepcopy(ts)
                our_ts.observation['goal'] = tf.convert_to_tensor(value=[way_g])
                action = agent.policy.action(our_ts)
              else:
                action = agent.policy.action(ts)
    
            else:
              action = agent.policy.action(ts) #flat agent without subgoals, directly takes primitive actions
    
            ts = env.step(action)
            step += 1
    
          obs_vec = np.array(obs_vec)
    
          plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
          plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                      color='red', s=200, label='start')
          plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                      color='green', s=200, label='end')
          plt.scatter([goal[0]], [goal[1]], marker='*',
                      color='green', s=200, label='goal')
          
          plt.title(title, fontsize=24)
          
          if use_lsgvp:
              if lsgvp_subgoals != []:
                  lsgvp_subgoals = np.asarray(lsgvp_subgoals)
                  plt.plot(lsgvp_subgoals[:, 0], lsgvp_subgoals[:, 1], 'y-s', alpha=1.0, label='waypoint')
    
    
        plt.show()
    
        flat_srate = sum(success[0])/len(success[0])
        lsgvp_srate = sum(success[2])/len(success[2])
    
        print("No subgoals success rate: ", flat_srate, '\n', "LSGVP success rate: ", lsgvp_srate)