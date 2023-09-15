"""GridWorld_PuddleWindy Domain."""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import super
from builtins import int
from future import standard_library
standard_library.install_aliases()
from builtins import range
from past.utils import old_div
import numpy as np
import os

from problems.rlpy.Tools.GeneralTools import plt, FONTSIZE, linearMap, __rlpy_location__, findElemArray1D, perms
from .Domain import Domain

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class GridWorld_PuddleWindy(Domain):

    """
    The GridWorld_PuddleWindy domain simulates a path-planning problem for a mobile robot
    in an environment with obstacles. The goal of the agent is to
    navigate from the starting point to the goal state.
    
    The map is loaded from a text file filled with numbers showing the map with the following
    coding for each cell:

    * 0: empty
    * 1: blocked
    * 2: start
    * 3: goal
    * 4: pit

    **STATE:**
    The Row and Column corresponding to the agent's location. \n
    **ACTIONS:**
    Four cardinal directions: up, down, left, right (given that
    the destination is not blocked or out of range). \n
    **TRANSITION:**
    There is 30% probability of failure for each move, in which case the action
    is replaced with a random action at each timestep. Otherwise the move succeeds
    and the agent moves in the intended direction. \n
    **REWARD:**
    The reward on each step is -.001 , except for actions
    that bring the agent to the goal with reward of +1.\n

    """

    map = start_state = goal = None
    # Used for graphics to show the domain
    agent_fig = upArrows_fig = downArrows_fig = leftArrows_fig = None
    rightArrows_fig = domain_fig = valueFunction_fig = None
    #: Number of rows and columns of the map
    ROWS = COLS = 0
    #: Reward constants
    GOAL_REWARD = 0
    PIT_REWARD = -1
    PUDDLE_REWARD = -100
    BLOCK_REWARD = -1
    STEP_REWARD = -1
    discount_factor = 1
    #: Set by the domain = min(100,rows*cols)
    episodeCap = None
    #: Movement Noise
    NOISE = 0
    # Used for graphical normalization
    MAX_RETURN = 1
    RMAX = MAX_RETURN
    # Used for graphical normalization
    MIN_RETURN = -1
    # Used for graphical shifting of arrows
    SHIFT = .1

    actions_num = 4
    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, PUDDLE = list(range(6))
    
    #: Up, Down, Left, Right
    ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
    # directory of maps shipped with rlpy
    default_map_dir = os.path.join(
        __rlpy_location__,
        "Domains",
        "GridWorldMaps")

    def __init__(self, mapname, noise=.1, episodeCap=1000, WINDY=np.array([0,0,0,1,1,1,2,2,1,0])):
        self.map = np.loadtxt(mapname, dtype=np.uint8)
        if self.map.ndim == 1: self.map = self.map[np.newaxis, :]
        self.start_state = np.argwhere(self.map == self.START)[0]
        self.ROWS, self.COLS = np.shape(self.map)
        self.WINDY = WINDY

        self.statespace_limits = np.array([[0, self.ROWS - 1], [0, self.COLS - 1]])
        self.NOISE = noise
        self.DimNames = ['Row', 'Col']
        # 2*self.ROWS*self.COLS, small values can cause problem for some
        # planning techniques
        self.episodeCap = episodeCap
        super(GridWorld_PuddleWindy, self).__init__()

    def allStates(self):
        if self.continuous_dims == []:
            # Recall that discrete dimensions are assumed to be integer
            return (perms(self.discrete_statespace_limits[:,1] - self.discrete_statespace_limits[:,0] + 1) + 
                    self.discrete_statespace_limits[:,0])

    def showDomain(self, a=0, s=None):
        if s is None:
            s = self.state

        # Draw the environment
        if self.domain_fig is None:
            self.agent_fig = plt.figure("Domain")
            self.domain_fig = plt.imshow(self.map, cmap='GridWorld', interpolation='nearest', vmin=0, vmax=5)
            plt.xticks(np.arange(self.COLS), fontsize=FONTSIZE)
            plt.yticks(np.arange(self.ROWS), fontsize=FONTSIZE)
            # pl.tight_layout()
            self.agent_fig = plt.gca().plot(s[1], s[0], 'kd', markersize=20.0 - self.COLS)
            plt.show()
        self.agent_fig.pop(0).remove()
        self.agent_fig = plt.figure("Domain")
        #mapcopy = copy(self.map)
        #mapcopy[s[0],s[1]] = self.AGENT
        # self.domain_fig.set_data(mapcopy)
        # Instead of '>' you can use 'D', 'o'
        self.agent_fig = plt.gca().plot(s[1], s[0], 'k>', markersize=20.0 - self.COLS)
        plt.figure("Domain").canvas.draw()
        plt.figure("Domain").canvas.flush_events()

    def showLearning(self, representation):
        if self.valueFunction_fig is None:
            plt.figure("Value Function")
            self.valueFunction_fig = plt.imshow(
                self.map,
                cmap='ValueFunction',
                interpolation='nearest',
                vmin=self.MIN_RETURN,
                vmax=self.MAX_RETURN)
            plt.xticks(np.arange(self.COLS), fontsize=12)
            plt.yticks(np.arange(self.ROWS), fontsize=12)
            # Create quivers for each action. 4 in total
            X = np.arange(self.ROWS) - self.SHIFT
            Y = np.arange(self.COLS)
            X, Y = np.meshgrid(X, Y)
            DX = DY = np.ones(X.shape)
            C = np.zeros(X.shape)
            C[0, 0] = 1  # Making sure C has both 0 and 1
            # length of arrow/width of bax. Less then 0.5 because each arrow is
            # offset, 0.4 looks nice but could be better/auto generated
            arrow_ratio = 0.4
            Max_Ratio_ArrowHead_to_ArrowLength = 0.25
            ARROW_WIDTH = 0.5 * Max_Ratio_ArrowHead_to_ArrowLength / 5.0
            self.upArrows_fig = plt.quiver(Y, X, DY, DX, C, units='y', cmap='Actions', scale_units="height",
                                           scale=old_div(self.ROWS, arrow_ratio), width=-1 * ARROW_WIDTH)
            self.upArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(self.ROWS) + self.SHIFT
            Y = np.arange(self.COLS)
            X, Y = np.meshgrid(X, Y)
            self.downArrows_fig = plt.quiver(Y, X, DY, DX, C, units='y', cmap='Actions', scale_units="height", 
                                             scale=old_div(self.ROWS, arrow_ratio), width=-1 * ARROW_WIDTH)
            self.downArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(self.ROWS)
            Y = np.arange(self.COLS) - self.SHIFT
            X, Y = np.meshgrid(X, Y)
            self.leftArrows_fig = plt.quiver(Y, X, DY, DX, C, units='x', cmap='Actions', scale_units="width", 
                                             scale=old_div(self.COLS, arrow_ratio), width=ARROW_WIDTH)
            self.leftArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(self.ROWS)
            Y = np.arange(self.COLS) + self.SHIFT
            X, Y = np.meshgrid(X, Y)
            self.rightArrows_fig = plt.quiver(Y, X, DY, DX, C, units='x', cmap='Actions', scale_units="width", 
                                              scale=old_div(self.COLS, arrow_ratio), width=ARROW_WIDTH)
            self.rightArrows_fig.set_clim(vmin=0, vmax=1)
            plt.show()
        plt.figure("Value Function")
        V = np.zeros((self.ROWS, self.COLS))
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        Mask = np.ones((self.COLS, self.ROWS, self.actions_num), dtype='bool')
        arrowSize = np.zeros((self.COLS, self.ROWS, self.actions_num), dtype='float')
        # 0 = suboptimal action, 1 = optimal action
        arrowColors = np.zeros((self.COLS, self.ROWS, self.actions_num), dtype='uint8')
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.map[r, c] == self.BLOCKED:
                    V[r, c] = self.BLOCK_REWARD
                if self.map[r, c] == self.GOAL:
                    V[r, c] = self.MAX_RETURN
                if self.map[r, c] == self.PIT:
                    V[r, c] = self.MIN_RETURN
                if self.map[r, c] == self.EMPTY or self.map[r, c] == self.START:
                    s = np.array([r, c])
                    As = self.possibleActions(s)
                    terminal = self.isTerminal(s)
                    Qs = representation.Qs(s, terminal)
                    bestA = representation.bestActions(s, terminal, As)
                    V[r, c] = max(Qs[As])
                    Mask[c, r, As] = False
                    arrowColors[c, r, bestA] = 1

                    for i in range(len(As)):
                        a = As[i]
                        Q = Qs[i]
                        value = linearMap(Q, self.MIN_RETURN, self.MAX_RETURN, 0, 1)
                        arrowSize[c, r, a] = value
        
#        # write value function to txt file
#        with open('GridWorld_Flag3_Value.txt', 'a') as outfile:
#            outfile.write('\n' + str(np.round(V,decimals=2)) + '\n'*2)
            
        # Show Value Function
        self.valueFunction_fig.set_data(V)
        # Show Policy Up Arrows
        DX = arrowSize[:, :, 0]
        DY = np.zeros((self.ROWS, self.COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 0])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 0])
        C  = np.ma.masked_array(arrowColors[:, :, 0], mask=Mask[:,:, 0])
        self.upArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Down Arrows
        DX = -arrowSize[:, :, 1]
        DY = np.zeros((self.ROWS, self.COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 1])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 1])
        C  = np.ma.masked_array(arrowColors[:, :, 1], mask=Mask[:,:, 1])
        self.downArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Left Arrows
        DX = np.zeros((self.ROWS, self.COLS))
        DY = -arrowSize[:, :, 2]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 2])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 2])
        C  = np.ma.masked_array(arrowColors[:, :, 2], mask=Mask[:,:, 2])
        self.leftArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Right Arrows
        DX = np.zeros((self.ROWS, self.COLS))
        DY = arrowSize[:, :, 3]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 3])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 3])
        C  = np.ma.masked_array(arrowColors[:, :, 3], mask=Mask[:,:, 3])
        self.rightArrows_fig.set_UVC(DY, DX, C)
        plt.draw()

    def step(self, a):
        s = self.state
        wind = self.WINDY[s[1]]
        ns = self.state.copy()
        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())
        ns = self.state + self.ACTIONS[a]
        ns[0] = max(0, ns[0]-wind)
        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or
            ns[1] < 0 or ns[1] == self.COLS or
            self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
            r = self.BLOCK_REWARD
        else:
            # If in bounds, update the current state
            self.state = ns.copy()
            r = self.STEP_REWARD
            if self.map[ns[0], ns[1]] == self.PUDDLE:
                r += self.PUDDLE_REWARD
        if self.map[ns[0], ns[1]] == self.GOAL:
            r = self.GOAL_REWARD
        if self.map[ns[0], ns[1]] == self.PIT:
            r = self.PIT_REWARD
        terminal = self.isTerminal()
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self.start_state.copy()
        return self.state, self.isTerminal(), self.possibleActions()

    def isTerminal(self, s=None):
        if s is None:
            s = self.state
        if self.map[int(s[0]), int(s[1])] == self.GOAL:
            return True
        if self.map[int(s[0]), int(s[1])] == self.PIT:
            return True
        return False

    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        possibleA = np.array([], np.uint8)
        for a in range(self.actions_num):
            ns = s[0:2] + self.ACTIONS[a]
            if (ns[0] < 0 or ns[0] == self.ROWS or 
                ns[1] < 0 or ns[1] == self.COLS or 
                self.map[int(ns[0]), int(ns[1])] == self.BLOCKED):
                continue
            possibleA = np.append(possibleA, [a])
        return possibleA

    def expectedStep(self, s, a):
        # Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
        # pa: k-by-??   possible actions for each next state
        actions = self.possibleActions(s)
        k = len(actions)
        # Make Probabilities
        intended_action_index = findElemArray1D(a, actions)
        p = np.ones((k, 1)) * self.NOISE / (k * 1.)
        p[intended_action_index, 0] += 1 - self.NOISE
        # Make next states
        ns = np.tile(s, (k, 1)).astype(int)
        actions = self.ACTIONS[actions]
        ns += actions
        # Make next possible actions
        pa = np.array([self.possibleActions(sn) for sn in ns])
        # Make rewards
        r = np.ones((k, 1)) * self.STEP_REWARD
        goal = self.map[ns[:, 0].astype(np.int), ns[:, 1].astype(np.int)] == self.GOAL
        pit = self.map[ns[:, 0].astype(np.int), ns[:, 1].astype(np.int)] == self.PIT
        r[goal] = self.GOAL_REWARD
        r[pit] = self.PIT_REWARD
        # Make terminals
        t = np.zeros((k, 1), bool)
        t[goal] = True
        t[pit] = True
        return p, r, ns, t, pa

    

class GridWorld_PuddleWindy_Flag(GridWorld_PuddleWindy):

    def __init__(self, mapname, noise=.1, episodeCap=1000, 
                 WINDY = np.array([0,0,0,1,1,1,2,2,1,0]), 
                 FlagPos=np.array([[0,0]]), 
                 FlagWid=np.array([[1]]), 
                 FlagHeight=np.array([[1]])):
        super(GridWorld_PuddleWindy_Flag, self).__init__(mapname=mapname, noise=noise, episodeCap=episodeCap, WINDY=WINDY)
        self.collectedFlags = 0
        self.FlagPos = FlagPos
        self.FlagWid = FlagWid
        self.FlagHeight = FlagHeight
        self.FlagNum = FlagPos.shape[0]
        
        self.STEP_REWARD = np.zeros((self.ROWS, self.COLS, self.ROWS, self.COLS, self.FlagNum+1))
        self.REWARD = np.zeros((self.ROWS, self.COLS, self.ROWS, self.COLS, self.FlagNum+1))
        for flag in range(self.FlagNum):
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    s = np.array([[r, c]])
                    exponent = np.divide(np.sum(0.5*(s-self.FlagPos[flag,:])**2, axis=1), self.FlagWid[flag])
                    phi_s = np.multiply(self.FlagHeight[flag], np.exp(-exponent))
                    for nr in range(self.ROWS):
                        for nc in range(self.COLS):
                            ns = np.array([[nr, nc]])
                            exponentNext = np.divide(np.sum(0.5*(ns-self.FlagPos[flag,:])**2, axis=1), self.FlagWid[flag])
                            phi_ns = np.multiply(self.FlagHeight[flag], np.exp(-exponentNext))
                            self.STEP_REWARD[r,c,nr,nc,flag] = self.discount_factor * phi_ns - phi_s
                            self.REWARD[r,c,nr,nc,flag] = self.STEP_REWARD[r,c,nr,nc,flag]
                            if self.map[nr, nc] == self.GOAL:
                                self.REWARD[r,c,nr,nc,flag] = self.GOAL_REWARD + self.STEP_REWARD[r,c,nr,nc,flag]
                            if self.map[nr, nc] == self.PIT:
                                self.REWARD[r,c,nr,nc,flag] = self.PIT_REWARD + self.STEP_REWARD[r,c,nr,nc,flag]

        flag = self.FlagNum
        for r in range(self.ROWS):
            for c in range(self.COLS):
                s = np.array([[r, c]])
                for nr in range(self.ROWS):
                    for nc in range(self.COLS):
                        ns = np.array([[nr, nc]])
                        self.STEP_REWARD[r,c,nr,nc,flag] = 0
                        self.REWARD[r,c,nr,nc,flag] = self.STEP_REWARD[r,c,nr,nc,flag]
                        if self.map[nr, nc] == self.GOAL:
                            self.REWARD[r,c,nr,nc,flag] = self.GOAL_REWARD + self.STEP_REWARD[r,c,nr,nc,flag]
                        if self.map[nr, nc] == self.PIT:
                            self.REWARD[r,c,nr,nc,flag] = self.PIT_REWARD + self.STEP_REWARD[r,c,nr,nc,flag]

    def step(self, a):
        s = self.state
        wind = self.WINDY[s[1]]
        ns = self.state.copy()
        if self.random_state.random_sample() < self.NOISE:
            # Random Move
            a = self.random_state.choice(self.possibleActions())
        ns = self.state + self.ACTIONS[a]
        ns[0] = max(0, ns[0]-wind)
        
        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or
            ns[1] < 0 or ns[1] == self.COLS or
            self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
            r = self.BLOCK_REWARD + self.REWARD[s[0], s[1], ns[0], ns[1], self.collectedFlags]
        else:
            # If in bounds, update the current state
            self.state = ns.copy()
            r = self.REWARD[s[0], s[1], ns[0], ns[1], self.collectedFlags]
            if self.map[ns[0], ns[1]] == self.PUDDLE:
                r += self.PUDDLE_REWARD
        
        # update the collect flags
        for flag in range(self.FlagNum):
            if (ns[0] == int(self.FlagPos[flag,0]) and ns[1] == int(self.FlagPos[flag,1]) and self.collectedFlags == flag):
                self.collectedFlags = flag+1

        terminal = self.isTerminal()
        
        return r, ns, terminal, self.possibleActions()

    def s0(self):
        self.state = self.start_state.copy()
        self.collectedFlags = 0
        return self.state, self.isTerminal(), self.possibleActions()

    def expectedStep(self, s, a):
        # Returns k possible outcomes
        #  p: k-by-1    probability of each transition
        #  r: k-by-1    rewards
        # ns: k-by-|s|  next state
        #  t: k-by-1    terminal values
        # pa: k-by-??   possible actions for each next state
        actions = self.possibleActions(s)
        k = len(actions)
        # Make Probabilities
        intended_action_index = findElemArray1D(a, actions)
        p = np.ones((k, 1)) * self.NOISE / (k * 1.)
        p[intended_action_index, 0] += 1 - self.NOISE
        # Make next states
        ns = np.tile(s, (k, 1)).astype(int)
        actions = self.ACTIONS[actions]
        ns += actions
        # Make next possible actions
        pa = np.array([self.possibleActions(sn) for sn in ns])
        # Make rewards
        r = np.ones((k, 1))
        for i in range(k):
            r[i] = self.REWARD[s[0], s[1], ns[i,0], ns[i,1], self.collectedFlags]
        goal = self.map[ns[:,0].astype(np.int), ns[:,1].astype(np.int)] == self.GOAL
        pit = self.map[ns[:,0].astype(np.int), ns[:,1].astype(np.int)] == self.PIT
        blocked = self.map[ns[:,0].astype(np.int), ns[:,1].astype(np.int)] == self.BLOCKED
        r[blocked] += self.BLOCK_REWARD
        
        for nr in ns[:,0]:
            for nc in ns[:,1]:
                if nr < 0 or nr == self.ROWS or nc < 0 or nc == self.COLS:
                    r[nr, nc] += self.BLOCK_REWARD
        
        # Make terminals
        t = np.zeros((k, 1), bool)
        t[goal] = True
        t[pit] = True
        return p, r, ns, t, pa