"""GridWorld_Items Domain."""
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
from .GridWorld import GridWorld_Parent

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


#============================================================================================#
#||========================================================================================||#
#||                              grid-world with sub-goals                                 ||#
#||========================================================================================||#
#============================================================================================#
class GridWorld_Items(GridWorld_Parent):
    """
    the door is open after got the key
    """
    #: Reward constants
    GOAL_REWARD = +10
    SUBGOAL_REWARD = +10
    PIT_REWARD = -1
    BLOCK_REWARD = 0 # -0.1
    STEP_REWARD = 0
    discount_factor = 1
    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT, PUDDLE, DOOR, KEY = list(range(9))

    def __init__(self, mapname, noise=.1, episodeCap=1000, items_pos=None):

        self.items_pos = items_pos
        self.num_items = self.items_pos.shape[0]

        self.map = np.loadtxt(mapname, dtype=np.uint8)
        if self.map.ndim == 1: self.map = self.map[np.newaxis, :]
        self.ROWS, self.COLS = np.shape(self.map)

        self.start_state = np.argwhere(self.map == self.START)[0]
        self.start_state = np.hstack(( self.start_state, np.zeros((self.num_items,), dtype=np.int) ))

        self.statespace_limits = np.array([[0, self.ROWS-1], [0, self.COLS-1]])
        for _ in range(self.num_items):
            self.statespace_limits = np.vstack((self.statespace_limits, [0,1]))

        self.ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
        self.ACTIONS = np.hstack(( self.ACTIONS, np.zeros((4,self.num_items), dtype=np.int) ))

        super(GridWorld_Items, self).__init__(noise=noise, episodeCap=episodeCap)

    def s0(self):
        # print('start of an episode')
        self.state = self.start_state.copy()
        return self.state, self.isTerminal(), self.possibleActions()

    def step(self, a):
        if self.random_state.random_sample() < self.NOISE: # Random Move
            a = self.random_state.choice(self.possibleActions())
        ns = self.state + self.ACTIONS[a] # Take action
        r = self.STEP_REWARD

        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or 
            ns[1] < 0 or ns[1] == self.COLS or 
            self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
            r += self.BLOCK_REWARD

        # Compute the reward and update the last several dimensions of ns
        dim = 1
        for item_pos in self.items_pos:
            dim += 1
            if ( np.absolute(ns[0]-item_pos[0]) <= 0.5 and \
                    np.absolute(ns[1]-item_pos[1]) <= 0.5 and \
                    ns[dim] < 0.5 ):
                ns[dim] = 1
                r += self.SUBGOAL_REWARD
                break
        self.state = ns.copy()

        if self.map[ns[0], ns[1]] == self.GOAL:
            r += self.GOAL_REWARD
        if self.map[ns[0], ns[1]] == self.PIT:
            r += self.PIT_REWARD

        terminal = self.isTerminal()
        return r, ns, terminal, self.possibleActions()



#============================================================================================#
#||========================================================================================||#
#||                          grid-world with sub-goals and flags                           ||#
#||========================================================================================||#
#============================================================================================#
class GridWorld_Items_Flag(GridWorld_Parent):
    #: Reward constants
    GOAL_REWARD = +10
    SUBGOAL_REWARD = +10
    PIT_REWARD = -1
    BLOCK_REWARD = 0 # -0.1
    STEP_REWARD = 0
    discount_factor = 1
    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT, PUDDLE, DOOR, KEY = list(range(9))

    def __init__(self, mapname, noise=.1, episodeCap=1000, items_pos=None, 
                 FlagPos=np.array([[0,0]]), 
                 FlagWid=np.array([[10]]), 
                 FlagHeight=np.array([[0.0]])):

        self.FlagPos = FlagPos
        self.FlagWid = FlagWid
        self.FlagHeight = FlagHeight
        self.FlagNum = FlagPos.shape[0]
        self.collectedFlags = 0

        self.items_pos = items_pos
        self.num_items = self.items_pos.shape[0]

        self.map = np.loadtxt(mapname, dtype=np.uint8)
        if self.map.ndim == 1: self.map = self.map[np.newaxis, :]
        self.ROWS, self.COLS = np.shape(self.map)
        
        self.start_state = np.argwhere(self.map == self.START)[0]
        self.start_state = np.hstack(( self.start_state, np.zeros((self.num_items,), dtype=np.int) ))
        self.start_state = np.hstack(( self.start_state, [0] ))

        self.statespace_limits = np.array([[0, self.ROWS-1], [0, self.COLS-1]])
        for _ in range(self.num_items):
            self.statespace_limits = np.vstack((self.statespace_limits, [0,1]))
        self.statespace_limits = np.vstack(( self.statespace_limits, [0,self.FlagNum] ))

        self.ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
        self.ACTIONS = np.hstack(( self.ACTIONS, np.zeros((4,self.num_items), dtype=np.int) ))
        self.ACTIONS = np.hstack(( self.ACTIONS, np.zeros((4,1), dtype=np.int) ))

        super(GridWorld_Items_Flag, self).__init__(noise=noise, episodeCap=episodeCap)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(self.ACTIONS)
        # print(self.start_state)
        # print(self.statespace_limits)
        
        self.STEP_REWARD = 0.0 * np.ones((self.ROWS, self.COLS, self.ROWS, self.COLS, self.FlagNum+1))
        for flag in range(self.FlagNum):
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    s_pos = np.array([[r, c]])
                    exponent = np.divide(np.sum(0.5*(s_pos-self.FlagPos[flag,:])**2, axis=1), self.FlagWid[flag])
                    phi_s = np.multiply(self.FlagHeight[flag], np.exp(-exponent))
                    for nr in range(self.ROWS):
                        for nc in range(self.COLS):
                            ns_pos = np.array([[nr, nc]])
                            exponentNext = np.divide(np.sum(0.5*(ns_pos-self.FlagPos[flag,:])**2, axis=1), self.FlagWid[flag])
                            phi_ns = np.multiply(self.FlagHeight[flag], np.exp(-exponentNext))
                            self.STEP_REWARD[r,c,nr,nc,flag] += self.discount_factor * phi_ns - phi_s

    def s0(self):
        # print('start of an episode')
        self.state = self.start_state.copy()
        self.collectedFlags = 0
        return self.state, self.isTerminal(), self.possibleActions()
        
    def step(self, a):
        s = self.state
        collectedFlag = s[-1]
        if self.random_state.random_sample() < self.NOISE: # Random Move
            a = self.random_state.choice(self.possibleActions())
        ns = self.state + self.ACTIONS[a] # Take action
        r = self.STEP_REWARD[s[0], s[1], ns[0], ns[1], collectedFlag]

        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or 
            ns[1] < 0 or ns[1] == self.COLS or 
            self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
            r += self.BLOCK_REWARD
        
        # Compute the reward and update the last several dimensions of ns
        dim = 1
        for item_pos in self.items_pos:
            dim += 1
            if ( np.absolute(ns[0]-item_pos[0]) <= 0.5 and \
                    np.absolute(ns[1]-item_pos[1]) <= 0.5 and \
                    ns[dim] < 0.5 ):
                ns[dim] = 1
                r += self.SUBGOAL_REWARD
                # print('subgoal', item_pos)
                break
        if self.map[ns[0], ns[1]] == self.GOAL:
            r += self.GOAL_REWARD
        if self.map[ns[0], ns[1]] == self.PIT:
            r += self.PIT_REWARD

        # update the last dimension of state space (number of collected flags)
        if ( collectedFlag < self.FlagNum and \
             np.absolute(ns[0] - self.FlagPos[collectedFlag,0]) <= 0.5 and \
             np.absolute(ns[1] - self.FlagPos[collectedFlag,1]) <= 0.5):
            # print(ns, r, 'collect flag', self.FlagPos[collectedFlag,:])
            collectedFlag += 1
        # else: print(ns, r)
        ns[-1] = collectedFlag
        self.collectedFlags = collectedFlag
        self.state = ns.copy()

        terminal = self.isTerminal()
        
        return r, ns, terminal, self.possibleActions()