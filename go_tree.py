import logging
from go_board import *
from game import *
from hyper_params import *
from model import *
import go_board

class GoTreeNode():
    def __init__(self, state=None, action_taken=None, prob=None):
        self.action_taken = action_taken        
        self.w = 0
        self.n = 0
        self.q = 0
        self.p = prob
        self.outcome = None
        self.state = state
        self.parent = None        
        self.actions = {}

    def add_value(self, value):
        self.w += value
        self.n += 1
        self.q = self.w / self.n      

    def calc_score(self):
        return self.q + C_PUCT*self.p*np.sqrt(1+self.parent.n)/(1+self.n)

    def add_child(self, action: int, child_node):
        child_node.parent = self
        self.actions[action] = child_node        


class GoPlayer():
    def __init__(self, starting_state, model: Model, noise_alpha=0.0, temperatures=None):
        self.root = GoTreeNode(state=starting_state, action_taken=None, prob=1.0)
        self.model = model
        self.noise_alpha = noise_alpha
        self.temperatures = temperatures    
        #expand the root
        self.run_mcts(1)

    def find_temperature(self, move_index):
        for start, end, temp in self.temperatures:
            if start<=move_index and move_index<end:
                return temp

    def opponent_played(self, move_index, action):        
        action_node = self.root.actions[action]
        self.ensure_node_state(action_node)
        self.make_root(action_node)
    
    def play(self, move_index):
        self.run_mcts(MCTS_STEPS)        
                
        temperature = self.find_temperature(move_index)
        act_distrib = self.count_probabilites(self.root, temperature)

        if self.noise_alpha > 0.0:
            # add Dir(0.03) dirichlet noise for additional exploration 
            dirichlet_alpha = np.zeros(ACTION_SPACE_SIZE, dtype=np.float)
            dirichlet_alpha[act_distrib>0.0] = self.noise_alpha                
            dirichlet_noise = np.random.dirichlet(alpha=dirichlet_alpha)
            act_distrib = (1.0-EPS)*act_distrib + EPS*dirichlet_noise
        
        # action distribution sanity check        
        assert np.abs(np.sum(act_distrib)-1.0)<1e-7, "Invalid act_distrib: sum=%.7f, n=%d, action_n:%s, distrib: %s" % (np.sum(act_distrib), tree.root.n, np.sum([act.n for act in tree.root.actions.values()]), act_distrib)
        assert np.product(act_distrib.shape) == (ACTION_SPACE_SIZE,)

        while True:
            action = np.random.choice(ACTION_SPACE_SIZE, p=act_distrib)
            action_node = self.root.actions[action]
            self.ensure_node_state(action_node)

            if action_node.state is None:
                # an invalid action node has been chosen, remove the node and try again
                logging.debug("Encountered invalid action: %d for player %d in position:\n%s", 
                                action, self.root.state.player, go_board.to_pretty_print(self.root.state.pos[-1]))
                logging.debug("act_distrib: %s", act_distrib)
                act_distrib[action] = 0.0
                act_distrib = act_distrib / np.sum(act_distrib)
                self.remove_node(action_node)                        
            else:
                self.make_root(action_node)
                break

        return act_distrib, action, self.root.q, self.root.outcome    

    def ensure_node_state(self, node):
        if node.state is not None:
            return
        node.state, node.outcome = go_board.next_state(node.parent.state, node.action_taken)

    def remove_node(self, node):
        del node.parent.actions[node.action_taken]
        node.parent = None

    def run_mcts(self, num_steps):        
        for step_index in np.arange(num_steps):
            # select
            leaf_node = self.select()
            # expand
            value = self.expand(leaf_node)

            # if this is an invalid state node, we remove it from the parent's valid actions
            if value is None:
                self.remove_node(leaf_node)                     
            else:                
                # backup
                self.backup(leaf_node, value)
        logging.debug("MCTS size: %d" % self.root.n)

    def count_probabilites(self, node: GoTreeNode, temperature):
        new_act_probabilities = np.zeros(ACTION_SPACE_SIZE, dtype=np.float)        
        if temperature<TEMPERATURE_MIN:
            max_action = max(node.actions, key=lambda k: node.actions[k].n)
            new_act_probabilities[max_action] = 1.0
        else:
            exp_action_counts = np.array([node.actions[act].n**(1/temperature) for act in node.actions.keys()])
            new_act_probabilities[list(node.actions.keys())] = exp_action_counts/np.sum(exp_action_counts)
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
        while len(node.actions) > 0:            
            max_action = max(node.actions, key=lambda k: node.actions[k].calc_score())
            node = node.actions[max_action]
        return node

    def expand(self, node: GoTreeNode):      
        self.ensure_node_state(node)        
        if node.state is None:
            # if the state is None, it's an invalid move, so we return value=None
            return None

        if node.outcome is not None:
            if node.outcome == OUTCOME_DRAW:
                value = REWARD_DRAW
            else:            
                if node.state.player==PLAYER_1 and node.outcome==OUTCOME_WIN_PLAYER_1:
                    value = REWARD_WIN
                else:
                    value = REWARD_LOOSE
        else:            
            augumented_state = go_board.sample_dihedral_transformation(node.state)
            value, actions_probalities = self.model.predict(augumented_state)        

            # expand all valid? actions
            valid_actions = go_board.maybe_valid_actions(node.state)
            for action in valid_actions:
                new_node = GoTreeNode(state=None, action_taken=action, prob=actions_probalities[action])
                node.add_child(action, new_node)

        assert value is not None
        return value


    def backup(self, node: GoTreeNode, value):
        current_node = node
        leaf_player = node.state.player
        while current_node:
            value_sign = 1.0 if leaf_player == current_node.state.player else -1.0
            current_node.add_value(value_sign * value)
            current_node = current_node.parent


    