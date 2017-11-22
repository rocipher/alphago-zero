import logging
from go_board import *
from game import *
from hyper_params import *
from model import *

class GoTreeNode():
    def __init__(self, state, prob=None, outcome=None):
        self.state = state
        self.outcome = outcome
        self.w = 0
        self.n = 0
        self.q = 0
        self.p = prob
        self.parent = None        
        self.actions = {}
        self.score = None

    def add_value(self, value):
        self.w += value
        self.n += 1
        self.q = self.w / self.n
        if self.parent is not None:
            self.score = self.calc_score()

    def calc_score(self):
        return self.q + C_PUCT*self.p*np.sqrt(self.parent.n)/(1+self.n)

    def add_child(self, action: int, child_node):
        child_node.parent = self
        child_node.score = child_node.calc_score()
        self.actions[action] = child_node        


class GoPlayer():
    def __init__(self, starting_state, model: Model, noise_alpha=0.0, temperatures=None):
        self.root = GoTreeNode(starting_state)
        self.model = model
        self.noise_alpha = noise_alpha
        self.temperatures = temperatures       

    def find_temperature(self, move_index):
        for start, end, temp in self.temperatures:
            if start<=move_index and move_index<end:
                return temp

    def opponent_played(self, move_index, action):
        self.run_mcts(MCTS_STEPS)
        action_node = self.root.actions[action]
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

        action = np.random.choice(ACTION_SPACE_SIZE, p=act_distrib)
            
        action_node = self.root.actions[action]
        self.make_root(action_node)

        return act_distrib, action, self.root.outcome      

    def run_mcts(self, num_steps):        
        for step_index in np.arange(num_steps):
            # select
            leaf_node = self.select()
            # expand
            value = self.expand(leaf_node)
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
            max_action = max(node.actions, key=lambda k: node.actions[k].score)
            node = node.actions[max_action]
        return node

    def expand(self, node: GoTreeNode):
        augumented_state = GoBoard.sample_dihedral_transformation(node.state)
        value, actions_probalities = self.model.predict(augumented_state)

        # expand all valid actions
        valid_actions = GoBoard.valid_actions(node.state)
        for action in valid_actions:
            child_state, outcome = GoBoard.next_state(node.state, action)
            new_node = GoTreeNode(child_state, actions_probalities[action], outcome)
            node.add_child(action, new_node)

        return value


    def backup(self, node: GoTreeNode, value):
        current_node = node
        leaf_player = node.state.player
        while current_node:
            value_sign = 1.0 if leaf_player == current_node.state.player else -1.0
            current_node.add_value(value_sign * value)
            current_node = current_node.parent


    