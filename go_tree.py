import logging
from go_board import *
from game import *
from hyper_params import *
from model import *
import go_board

class GoTreeNode():
    def __init__(self, state=None, action_taken=None, action_prob_distrib=None, outcome=None):
        self.action_taken = action_taken        
        self.w = np.zeros(ACTION_SPACE_SIZE, dtype=float)
        self.n = np.zeros(ACTION_SPACE_SIZE, dtype=int)
        self.q = np.zeros(ACTION_SPACE_SIZE, dtype=float)
        self.p = action_prob_distrib
        self.sum_n = 0
        self.outcome = outcome
        self.state = state
        self.parent = None        
        self.actions = {}

    def add_value(self, action, value):
        self.w[action] += value
        self.n[action] += 1
        self.sum_n += 1
        self.q[action] = self.w[action] / self.n[action]

    def puct_distrib(self):
        return self.q + C_PUCT*self.p*np.sqrt(self.sum_n)/(1+self.n)

    def add_child(self, action: int, child_node):
        child_node.parent = self
        self.actions[action] = child_node        


class GoPlayer():
    def __init__(self, starting_state, model: Model, noise_alpha=0.0, temperatures=None):
        self.model = model
        _, actions_probalities = self.model.predict(starting_state)
        self.root = GoTreeNode(state=starting_state, action_taken=None, 
                               action_prob_distrib=actions_probalities, outcome=None)                
        self.noise_alpha = noise_alpha
        self.temperatures = temperatures    

    def find_temperature(self, move_index):
        for start, end, temp in self.temperatures:
            if start<=move_index and move_index<end:
                return temp

    def opponent_played(self, move_index, action):                
        if action not in self.root.actions:
            action_node, _ = self.expand(self.root, action)
        else:
            action_node = self.root.actions[action]
        self.make_root(action_node)
    
    @staticmethod
    def vector_pretty_print(v):
        return "[" + " ".join(["%.2f" % p for p in v]) + "]"

    def play(self, move_index):
        self.run_mcts(MCTS_STEPS)        
                
        temperature = self.find_temperature(move_index)
        act_distrib = self.count_probabilites(self.root, temperature)

        logging.debug("Move index: %d\nn: %s\np: %s\nq: %s\npuct: %s",
                     move_index, self.root.n, 
                     GoPlayer.vector_pretty_print(self.root.p), 
                     GoPlayer.vector_pretty_print(self.root.q), 
                     GoPlayer.vector_pretty_print(self.root.puct_distrib()))

        if self.noise_alpha > 0.0:
            # add Dir(0.03) dirichlet noise for additional exploration 
            dirichlet_alpha = np.repeat(self.noise_alpha, ACTION_SPACE_SIZE)
            dirichlet_noise = np.random.dirichlet(alpha=dirichlet_alpha)
            act_distrib = (1.0-ACT_NOISE_EPS)*act_distrib + ACT_NOISE_EPS*dirichlet_noise
        
        # action distribution sanity check        
        assert np.abs(np.sum(act_distrib)-1.0)<1e-7, "Invalid act_distrib: sum=%.7f, n=%d, action_n:%s, distrib: %s" % (np.sum(act_distrib), tree.root.n, np.sum([act.n for act in tree.root.actions.values()]), act_distrib)
        assert np.product(act_distrib.shape) == (ACTION_SPACE_SIZE,)
        
        while True:
            action = np.random.choice(ACTION_SPACE_SIZE, p=act_distrib)
            q_action = self.root.q[action]
            if action not in self.root.actions:
                action_node, _ = self.expand(self.root, action)
            else:
                action_node = self.root.actions[action]
            
            if action_node is None:
                # an invalid action node has been chosen, remove the node and try again
                # logging.debug("Encountered invalid action: %d for player %d in position:\n%s", 
                #                 action, self.root.state.player, go_board.to_pretty_print(self.root.state.pos[-1]))
                #logging.debug("act_distrib: %s", act_distrib)
                act_distrib[action] = 0.0
                act_distrib = act_distrib / np.sum(act_distrib)                     
            else:
                self.make_root(action_node)
                break

        return act_distrib, action, q_action, self.root.outcome    


    def run_mcts(self, num_steps):       
        #logging.debug("MCTS add nodes: %d", num_steps-self.root.sum_n)

        nodes_added = 0
        while nodes_added<num_steps:
        #for step_index in np.arange(num_steps):
            # select
            leaf_node, new_action = self.select()
            # expand
            new_node, value = self.expand(leaf_node, new_action)

            # if this is an invalid state node, we remove it from the parent's valid actions
            if value is None:
                # invalidate this action
                leaf_node.p[new_action] = 0.0
                # renormalize
                leaf_node.p = leaf_node.p / np.sum(leaf_node.p)
                assert leaf_node.n[new_action] == 0
                leaf_node.w[new_action] = -np.inf
                leaf_node.q[new_action] = -np.inf
            else:
                nodes_added += 1
                # backup
                self.backup(new_node, value)        

        logging.debug("MCTS size: %d", self.root.sum_n)

    def count_probabilites(self, node: GoTreeNode, temperature):
        new_act_probabilities = np.zeros(ACTION_SPACE_SIZE, dtype=np.float)        
        if temperature<TEMPERATURE_MIN:
            max_action = np.argmax(node.n)
            new_act_probabilities[max_action] = 1.0
        else:                        
            if temperature == 1.0:
                # optimization for the common case when temperature equals 1.0
                new_act_probabilities = node.n/node.sum_n
            else:
                exp_action_counts = node.n**(1/temperature)
                new_act_probabilities = exp_action_counts/np.sum(exp_action_counts)
        return new_act_probabilities


    def make_root(self, node: GoTreeNode):
        old_root = self.root
        node.parent = None
        self.root = node
        del old_root


    def select(self):
        # choose the action that maximizes Q(s, a) + U(s, a)
        # where U(s, a) = Cpuct*P(s, a)sum_over_b(N(s, b))/(1+N(s, a)))
        node = self.root
        max_puct_action = None
        while True:
            if node.outcome:
                max_puct_action = None
                break
            max_puct_action = np.argmax(node.puct_distrib())
            if max_puct_action in node.actions:
                node = node.actions[max_puct_action]
            else:
                break
        return node, max_puct_action


    def get_reward(self, outcome, player):
        if outcome == OUTCOME_DRAW:
            return REWARD_DRAW
        else:            
            if player==PLAYER_1 and outcome==OUTCOME_WIN_PLAYER_1:
                return REWARD_WIN
            elif player==PLAYER_2 and outcome==OUTCOME_WIN_PLAYER_2:
                return REWARD_WIN
            else:
                return REWARD_LOOSE
    

    def expand(self, node: GoTreeNode, action):
        if node.outcome is not None:
            # this is already an end node, return it's value
            value = self.get_reward(node.outcome, node.parent.state.player)
            return node, value

        new_node_state, outcome = go_board.next_state(node.state, action)
        if new_node_state is None:
            # if the state is None, it's an invalid move, so we return value=None
            return None, None

        augumented_state, inverse_transform_func = go_board.sample_dihedral_transformation(new_node_state)
        value, actions_probalities_inversed = self.model.predict(augumented_state)
        pass_prob = actions_probalities_inversed[ACTION_SPACE_SIZE-1]
        act_prob_inv_mat = actions_probalities_inversed[:BOARD_SIZE*BOARD_SIZE].reshape(BOARD_SIZE, BOARD_SIZE)        
        actions_probalities = np.concatenate([inverse_transform_func(act_prob_inv_mat).flatten(), [pass_prob]])
        
        assert len(actions_probalities) == ACTION_SPACE_SIZE
        assert np.abs(np.sum(actions_probalities)-1.0)<1e-4

        if outcome is not None:
            value = self.get_reward(outcome, node.state.player)
        new_node = GoTreeNode(state=new_node_state, action_taken=action, 
                              action_prob_distrib=actions_probalities, outcome=outcome)
        node.add_child(action, new_node)
        assert value is not None
        return new_node, value


    def backup(self, node: GoTreeNode, value):
        child_node = node
        leaf_player = child_node.parent.state.player
        while child_node.parent:
            value_sign = 1.0 if leaf_player == child_node.parent.state.player else -1.0
            child_node.parent.add_value(child_node.action_taken, value_sign * value)
            child_node = child_node.parent


    