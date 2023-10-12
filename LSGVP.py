import copy
import itertools
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
import random
import networkx as nx
import tqdm
from tf_agents.trajectories import time_step

from envs.twoD_mazes import plot_walls

class LSGVP_Model:
    
    def __init__(self):
        
        self.subgoal_graph = None
        self.subgoal_to_node = {}
        self.node_to_subgoal = {}
        self.known_paths = {}
        self.terminal_subgoal = {}
        self.subgoal_node_label = 0
        self.subgoal_graph_updates = 0
        
        self.state_repo = []
        self.env = None
        self.actor = None
        self.critic = None
        
        self.pairwise_distances = {}
        self.merger_valuediff_limit = 7
        self.edge_distance_cutoff = 20
        
    
    #### display the subgoal graph
    def display_spatial_graph(self, G, node_to_state, other_points=None):
        
        edges = list(G.edges)
        
        plt.figure(figsize=(6, 6))
        plot_walls(self.env.pyenv.envs[0].env.walls)
        
        for e in edges:
              s_i = node_to_state[e[0]]
              s_j = node_to_state[e[1]]
              plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
        
        if other_points is not None:
          for p in other_points:
            print(p)
            plt.scatter([p[0]], [p[1]], marker='*',
                  color=p[2], s=200)
        plt.show()
          
    #get distance between two states or points, one can be a subgoal or goal
    def get_predicted_distance(self, obs_tensor_in, goal_tensor_in=None, aggregate='mean', masked=False):
        
        obs_tensor = copy.deepcopy(obs_tensor_in)

        if goal_tensor_in is None:
          goal_tensor = obs_tensor
        else:
          goal_tensor = copy.deepcopy(goal_tensor_in)

        for i in range(len(obs_tensor)):
          obs_tensor[i] = np.asarray(obs_tensor[i])
        obs_tensor = np.asarray(obs_tensor)

        for i in range(len(goal_tensor)):
          goal_tensor[i] = np.asarray(goal_tensor[i])
        goal_tensor = np.asarray(goal_tensor)

        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
          obs = obs_tensor[obs_index]
          obs_repeat_tensor = tf.ones_like(goal_tensor) * tf.expand_dims(obs, 0)
          obs_goal_tensor = {'observation': obs_repeat_tensor,
                            'goal': goal_tensor}
          pseudo_next_time_steps = time_step.transition(obs_goal_tensor,
                                                        reward=0.0,  # Ignored
                                                        discount=1.0)
          dist = self.actor._get_dist_to_goal(pseudo_next_time_steps, aggregate=aggregate) #distance function based on Q-values
          dist_matrix.append(dist)

        pairwise_dist = tf.stack(dist_matrix)
        if aggregate is None:
          pairwise_dist = tf.transpose(a=pairwise_dist, perm=[1, 0, 2])

        if masked:
          mask = (pairwise_dist > self._max_search_steps)
          return tf.compat.v1.where(mask, tf.fill(pairwise_dist.shape, np.inf), 
                            pairwise_dist)
        else:
          return pairwise_dist

    
    def graph_abstraction(self):

        ''' GRAPH ABSTRACTION '''  
        start = time.time()
        ret = self.get_predicted_distance(self.state_repo, aggregate=None)
        ret = np.max(ret, axis=0)
        end = time.time()
        print('Distance calc time: ', end-start)

        start = time.time()
        init_pivot = 0
        pivots = [init_pivot]
        subgoal_states = [self.state_repo[init_pivot]]

        proxy_paths = list(itertools.permutations(range(len(self.state_repo)), 2))
        proxy_paths = random.sample(proxy_paths, 1000)

        #proxy_goals = random.sample(range(len(self.state_repo)), 500)
        #proxy_starts = random.sample(range(len(self.state_repo)), 500)

        for i, s_i in enumerate(self.state_repo):
          unique = 1
          #print(pivots)
          for j in pivots:
            equi = 1
            self.pairwise_distances[(tuple(s_i), tuple(self.state_repo[j]))] = ret[i, j]
            self.pairwise_distances[(tuple(self.state_repo[j]), tuple(s_i))] = ret[j, i]
            
            cost_thru_i = []
            cost_thru_j = []
            for pp in proxy_paths:
              clip = np.inf
              cost_thru_i.append(min(clip, ret[pp[0]][i] + ret[i][pp[1]]))
              cost_thru_j.append(min(clip, ret[pp[0]][j] + ret[j][pp[1]]))
            cost_thru_i = np.asarray(cost_thru_i)
            cost_thru_j = np.asarray(cost_thru_j)
            if max(abs(cost_thru_i - cost_thru_j)) <= self.merger_valuediff_limit:
              pass
            else:
              equi = 0
            
            if equi == 1:
              unique = 0
              break

          if unique == 1:
            pivots.append(i)
            subgoal_states.append(s_i)
        end = time.time()
        print('Abstraction time: ', end-start)
        
        return subgoal_states


    def create_subgoal_graph(self, ret, edge_fraction): #### return the updated graph to the agent

        start = time.time()
        
        subgoal_states = ret 
        print('subgoal states: ', len(subgoal_states))
        
        for gs in subgoal_states:
            if tuple(gs) not in self.subgoal_to_node:
                self.subgoal_to_node[tuple(gs)] = self.subgoal_node_label
                self.node_to_subgoal[self.subgoal_node_label] = tuple(gs)
                self.subgoal_node_label += 1

        edge_distance_cutoff = self.edge_distance_cutoff
        while 1:
          print('cutoff: ', edge_distance_cutoff, self.edge_distance_cutoff)
          self.subgoal_graph = nx.DiGraph()
          subgoal_state_pairs = list(itertools.combinations(subgoal_states, 2))
          print('subgoal pairs: ', len(subgoal_state_pairs))
          
          for pair in subgoal_state_pairs:

              distance = self.pairwise_distances[(tuple(pair[0]), tuple(pair[1]))]
              if distance <= edge_distance_cutoff:
                self.subgoal_graph.add_edge(self.subgoal_to_node[tuple(pair[0])], self.subgoal_to_node[tuple(pair[1])], weight=distance)

              distance = self.pairwise_distances[(tuple(pair[1]), tuple(pair[0]))]
              if distance <= edge_distance_cutoff:
                self.subgoal_graph.add_edge(self.subgoal_to_node[tuple(pair[1])], self.subgoal_to_node[tuple(pair[0])], weight=distance)
    
          end = time.time()
          print('subgoal graph creation time: ', end-start)
          num_edges = len(list(self.subgoal_graph.edges))
          print('subgoal graph edges: ', num_edges)
          
          #this controls the sugoal graph density. If the distance cutoff is too small, we get a very sparse graph in which planning fail. So, we keep adjusting the distance cutoff according to the desired fraction of num. edges to total subgoal (node) pairs
          if num_edges/len(subgoal_state_pairs) < edge_fraction: 
            self.edge_distance_cutoff += 1
            edge_distance_cutoff = self.edge_distance_cutoff
          else:
            break

        self.display_spatial_graph(self.subgoal_graph, self.node_to_subgoal)
        self.subgoal_graph_updates += 1
    
    #find the subgoal node closest to the given state
    def localize(self, state, check_g=None, d_limit = None):
      
      local_g = None

      if d_limit is None:
        d_limit = 4

      if check_g is not None:
        check_g = copy.deepcopy(np.asarray(check_g))
        start_to_g_distances = self.get_predicted_distance([state], goal_tensor_in=[check_g], aggregate=None)
        start_to_g_distances = list(np.max(start_to_g_distances, axis=0)[0])

        dst = start_to_g_distances[0]
        min_dst = min(start_to_g_distances)
        if dst==min_dst and dst<=d_limit: ### target subgoal is reached only if its closest and within limit
          local_g = tuple(check_g)

      else:
        g_nodes = list(self.subgoal_graph.nodes)
        subgoals = np.asarray([np.asarray(self.node_to_subgoal[gn]) for gn in g_nodes])
        start_to_g_distances = self.get_predicted_distance([state], goal_tensor_in=subgoals, aggregate=None)
        start_to_g_distances = list(np.max(start_to_g_distances, axis=0)[0])
        
        min_d = np.inf
        min_idx = None
        for i in range(len(start_to_g_distances)):
          if start_to_g_distances[i] < min_d:
            min_d = start_to_g_distances[i]
            min_idx = i

        if start_to_g_distances[min_idx] <= d_limit:
          local_g = tuple(subgoals[min_idx])
      
      if local_g is not None and check_g is not None:
        if local_g != tuple(check_g):
          local_g = None

      return local_g
      
    #find subgoal path from given state to given goal, over the subgoal graph
    def plan(self, start, goal, min_edge=False):

      g_nodes = list(self.subgoal_graph.nodes)
      subgoals = np.asarray([np.asarray(self.node_to_subgoal[gn]) for gn in g_nodes])

      if subgoals != []:
        #print(len(list(self.subgoal_graph.edges)), len(self.subgoal_graph))
        start_to_g_distances = self.get_predicted_distance([start], goal_tensor_in=subgoals, aggregate=None)
        start_to_g_distances = np.max(start_to_g_distances, axis=0)[0]
        start_node = len(self.subgoal_graph) + 1
        self.node_to_subgoal[start_node] = tuple(start)
        start_to_g_edges = []

        if min_edge:
          dlimit = min([dst for dst in start_to_g_distances])
          #print(dlimit)
        else:
          dlimit = self.edge_distance_cutoff

        for i in range(len(start_to_g_distances)):
          if start_to_g_distances[i] <= dlimit:
            start_to_g_edges.append((start_node, self.subgoal_to_node[tuple(subgoals[i])], start_to_g_distances[i]))
        

        g_to_goal_distances = list(self.get_predicted_distance(subgoals, goal_tensor_in=[goal], aggregate=None))
        g_to_goal_distances = np.max(g_to_goal_distances, axis=0)
        end_node = len(self.subgoal_graph) + 2
        self.node_to_subgoal[end_node] = tuple(goal)
        g_to_goal_edges = []

        if min_edge:
          dlimit = min([dst[0] for dst in g_to_goal_distances])
          #print(dlimit)
        else:
          dlimit = self.edge_distance_cutoff

        for i in range(len(g_to_goal_distances)):
          if g_to_goal_distances[i][0] <= dlimit:
            g_to_goal_edges.append((self.subgoal_to_node[tuple(subgoals[i])], end_node, g_to_goal_distances[i][0]))

        for i in range(len(start_to_g_edges)):
          self.subgoal_graph.add_edge(start_to_g_edges[i][0], start_to_g_edges[i][1], weight=start_to_g_edges[i][2])
        
        for i in range(len(g_to_goal_edges)):
          self.subgoal_graph.add_edge(g_to_goal_edges[i][0], g_to_goal_edges[i][1], weight=g_to_goal_edges[i][2])

        #print(len(list(self.subgoal_graph.edges)), len(self.subgoal_graph))

        '''if len(start_to_g_edges)>1 or len(g_to_goal_edges)>1:
          print(start_to_g_edges, '\n', g_to_goal_edges)
          sys.exit()'''

        node_path = None
        try:
              node_path = nx.dijkstra_path(self.subgoal_graph, start_node, end_node, weight='weight')
              if len(node_path) < 2:
                  node_path = None
        except:
              pass
        
        if node_path is not None:
            state_path = [self.node_to_subgoal[nd] for nd in node_path]

        #self.display_spatial_graph(self.subgoal_graph, self.node_to_subgoal, 
         #                          other_points=[tuple([start[0],start[1],'red']), tuple([goal[0],goal[1],'green'])])

        for i in range(len(start_to_g_edges)):
          self.subgoal_graph.remove_edge(start_to_g_edges[i][0], start_to_g_edges[i][1])
        
        for i in range(len(g_to_goal_edges)):
          self.subgoal_graph.remove_edge(g_to_goal_edges[i][0], g_to_goal_edges[i][1])
        
        self.subgoal_graph.remove_node(start_node)
        self.subgoal_graph.remove_node(end_node)
        self.node_to_subgoal.pop(start_node, None)
        self.node_to_subgoal.pop(end_node, None)
        #print(len(list(self.subgoal_graph.edges)), len(self.subgoal_graph))
        
        if node_path is not None:
            return state_path
      
      return None
  
    ############################## Graph Pruning functions #############################
    def edge_testing(self, E, ts, steps, obs_vec, d_limit):
        keep = 1
        RE = 0
        i = 1
        test_subgoal = self.node_to_subgoal[E[1]]
        while i <= self.subgoal_edge_cutoff + 1:
          ts.observation['goal'] = tf.convert_to_tensor(value=[test_subgoal])
          obs_vec.append(ts.observation['observation'].numpy()[0])
          action = self.actor.policy.action(ts)
          RE += -1
          ts = self.env.step(action)
          steps += 1
          i += 1
      
          if self.localize(ts.observation['observation'].numpy()[0], d_limit=d_limit, check_g=test_subgoal) == tuple(test_subgoal):
            break
          
          if (0-RE) > self.subgoal_edge_cutoff:
            keep = 0 
        
        return keep, steps, obs_vec, ts

    def prune_graph(self, original_graph):
        
        #@title Graph Testing - 2
        test_iters = 1
        test_episodes = 300
        
        time_liniency = 0
        d_limit = 4
           
        self.env.pyenv.envs[0]._duration = 300
        #seed = np.random.randint(0, 1000000)
        self.env.pyenv.envs[0].gym.set_sample_goal_args(
            prob_constraint=0.0,
            min_dist=0,
            max_dist=np.inf)
        
        self.subgoal_graph = copy.deepcopy(original_graph)
        original_edges = list(self.subgoal_graph.edges)
        print('original edges: ', len(original_edges))
        
        unvisited_edges = [original_edges for _ in range(test_iters)]
        
        not_reach_cnt = 0
        reach_cnt = 0

        ts = self.env.reset()
        
        for iter in range(test_iters):
          obs_vec = []
          '''plt.figure(figsize=(12, 5))
          plot_walls(self.env.pyenv.envs[0].env.walls)'''
        
          for ep in tqdm.tnrange(test_episodes,
                                    desc='test progress'):
        
            local_g = None
            seek_g = None
            ts = self.env.reset()
        
            if unvisited_edges[iter] == []:
                break
        
            steps = 0
        
            while steps < self.env.pyenv.envs[0]._duration:
        
              local_g = self.localize(ts.observation['observation'].numpy()[0], d_limit=d_limit)
        
              if local_g is not None:
                #print('localized')
                local_g_node = self.subgoal_to_node[local_g]
                forth_edges = []
                ngbrs = list(self.subgoal_graph.neighbors(local_g_node))
                for ng in ngbrs:
                  if (local_g_node, ng) in unvisited_edges[iter]:
                    forth_edges.append((local_g_node, ng))
                
                if forth_edges != []:
                  seek_g = None
        
                for fe in forth_edges:
                  keep, steps, obs_vec, ts = self.edge_testing(fe, ts, steps, obs_vec, d_limit)
                  if keep == 0:
                    self.subgoal_graph.remove_edge(fe[0], fe[1])
        
                  unvisited_edges[iter].remove(fe)
        
                  '''if steps > eval_tf_env.pyenv.envs[0]._duration:
                    break'''
        
                  be = (fe[1], fe[0]) ## back edge
                  keep, steps, obs_vec, ts = self.edge_testing(be, ts, steps, obs_vec, d_limit)
                  if keep == 0:
                    if be in unvisited_edges[iter]:
                      self.subgoal_graph.remove_edge(be[0], be[1])
                      unvisited_edges[iter].remove(be)
                    
                    break #### didn't reach the source, so need to start fresh
                  
                  '''if steps > eval_tf_env.pyenv.envs[0]._duration:
                    break'''
        
                local_g = None
              
              if unvisited_edges[iter] == []:
                break
        
              if local_g == None:
                if seek_g is None:
                  ue = random.choice(unvisited_edges[iter])
                  random_subgoal = self.node_to_subgoal[ue[0]]
                  seek_g = [random_subgoal] * (2 * (self.subgoal_edge_cutoff + time_liniency)) ### use the same random target for a few steps
                
                ts.observation['goal'] = tf.convert_to_tensor(value=[seek_g[0]])
                seek_g.pop(0)
                if seek_g == []:
                  seek_g = None
        
                obs_vec.append(ts.observation['observation'].numpy()[0])
                action = self.actor.policy.action(ts)
                ts = self.env.step(action)
                steps += 1
        
            print(ep, steps, len(unvisited_edges[iter]))
        
          obs_vec = np.array(obs_vec)
          plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'o', alpha=0.3)
          plt.show()
        
        print(reach_cnt, not_reach_cnt, reach_cnt+not_reach_cnt)
        
        '''for iter in [0]:
          edges_to_remove[iter] = list(set(edges_to_remove[iter]))
          print(edges_to_remove[iter])
          for rme in edges_to_remove[iter]:
            self.subgoal_graph.remove_edge(rme[0],rme[1])'''
        
        print('isolates: ', list(nx.isolates(self.subgoal_graph)))
        self.subgoal_graph.remove_nodes_from(list(nx.isolates(self.subgoal_graph)))
        
        e = list(self.subgoal_graph.edges)
        print(len(e))
        #self.display_spatial_graph(self.subgoal_graph, self.node_to_subgoal)
        
        new_graph = copy.deepcopy(self.subgoal_graph)
        
        return new_graph
      

