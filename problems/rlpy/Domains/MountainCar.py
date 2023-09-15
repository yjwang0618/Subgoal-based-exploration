"""Classic mountain car task."""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from builtins import super
from future import standard_library
standard_library.install_aliases()
from problems.rlpy.Tools import plt, bound, fromAtoB
from problems.rlpy.Tools import lines
from .Domain import Domain
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Josh Joseph", "Alborz Geramifard"]



#============================================================================================#
#||========================================================================================||#
#||                            parent class for mountain-car                               ||#
#||========================================================================================||#
#============================================================================================#
class MountainCar_parent(Domain):

    """
    The goal is to drive an under accelerated car up to the hill.\n
    **STATE:**        Position and velocity of the car [x, xdot] \n
    **ACTIONS:**      [Acc backwards, Coast, Acc forward] \n
    **TRANSITIONS:**  Move along the hill with some noise on the movement. \n
    **REWARD:**       -1 per step and 0 at or beyond goal (``x-goal > 0``). \n
    There is optional noise on vehicle acceleration.
    **REFERENCE:**
    Based on `RL-Community Java Implementation <http://library.rl-community.org/wiki/Mountain_Car_(Java)>`_
    """

    actions_num = 3
    state_space_dims = 2
    continuous_dims = [0, 1]

    XMIN = -1.2  # : Lower bound on domain position
    XMAX = 0.6  #: Upper bound on domain position
    XDOTMIN = -0.07  # : Lower bound on car velocity
    XDOTMAX = 0.07  #: Upper bound on car velocity
    INIT_STATE = np.array([-0.5, 0.0])  # : Initial car state
    STEP_REWARD = -1  # : Penalty for each step taken before reaching the goal
    GOAL_REWARD = 0  #: Reward for reach the goal.
    #: X-Position of the goal location (Should be at/near hill peak)
    GOAL = .5
    actions = [-1, 0, 1]
    
    accelerationFactor = 0.001  # : Magnitude of acceleration action
    gravityFactor = -0.0025
    #: Hill peaks are generated as sinusoid; this is freq. of that sinusoid.
    hillPeakFrequency = 3.0

    # Used for visual stuff:
    domain_fig = None
    valueFunction_fig = None
    policy_fig = None
    actionArrow = None
    X_discretization = 20
    XDot_discretization = 20
    CAR_HEIGHT = .2
    CAR_WIDTH = .1
    ARROW_LENGTH = .2

    def __init__(self, start=-0.5, noise=0, discount_factor=0.9, episodeCap=1e4):
        """
        :param noise: Magnitude of noise (times accelerationFactor) in stochastic velocity changes
        """
        self.INIT_STATE[0] = start
        self.noise = noise #: Magnitude of noise (times accelerationFactor) in stochastic velocity changes
        self.discount_factor = discount_factor
        self.episodeCap = episodeCap
        # Visual stuff:
        self.xTicks = np.linspace(0, self.X_discretization - 1, 5)
        self.xTicksLabels = np.linspace(self.XMIN, self.XMAX, 5)
        self.yTicks = np.linspace(0, self.XDot_discretization - 1, 5)
        self.yTicksLabels = np.linspace(self.XDOTMIN, self.XDOTMAX, 5)
        self.MIN_RETURN     = self.STEP_REWARD * \
            (1 - self.discount_factor ** self.episodeCap) / \
            (1 - self.discount_factor) if self.discount_factor != 1 else self.STEP_REWARD * \
            self.episodeCap
        self.MAX_RETURN = 0
        self.DimNames = ['X', 'Xdot']
        super(MountainCar_parent, self).__init__()

    def step(self, a):
        """
        Take acceleration action *a*, adding noise as specified in ``__init__()``.
        """
        pass

    def s0(self):
        pass

    def isTerminal(self):
        """
        :return: ``True`` if the car has reached or exceeded the goal position.
        """
        return self.state[0] > self.GOAL

    def showDomain(self, a):
        s = self.state
        # Plot the car and an arrow indicating the direction of accelaration
        # Parts of this code was adopted from Jose Antonio Martin H.
        # <jamartinh@fdi.ucm.es> online source code
        pos, vel = s
        if self.domain_fig is None:  # Need to initialize the figure
            self.domain_fig = plt.figure("Mountain Car Domain")
            # plot mountain
            mountain_x = np.linspace(self.XMIN, self.XMAX, 1000)
            mountain_y = np.sin(3 * mountain_x)
            plt.gca(
            ).fill_between(mountain_x,
                           min(mountain_y) - self.CAR_HEIGHT * 2,
                           mountain_y,
                           color='g')
            plt.xlim([self.XMIN - .2, self.XMAX])
            plt.ylim(
                [min(mountain_y) - self.CAR_HEIGHT * 2,
                 max(mountain_y) + self.CAR_HEIGHT * 2])
            # plot car
            self.car = lines.Line2D([], [], linewidth=20, color='b', alpha=.8)
            plt.gca().add_line(self.car)
            # Goal
            plt.plot(self.GOAL, np.sin(3 * self.GOAL), 'yd', markersize=10.0)
            plt.axis('off')
            plt.gca().set_aspect('1')
        self.domain_fig = plt.figure("Mountain Car Domain")
        #pos = 0
        #a = 0
        car_middle_x = pos
        car_middle_y = np.sin(3 * pos)
        slope = np.arctan(3 * np.cos(3 * pos))
        car_back_x = car_middle_x - self.CAR_WIDTH * np.cos(slope) / 2.
        car_front_x = car_middle_x + self.CAR_WIDTH * np.cos(slope) / 2.
        car_back_y = car_middle_y - self.CAR_WIDTH * np.sin(slope) / 2.
        car_front_y = car_middle_y + self.CAR_WIDTH * np.sin(slope) / 2.
        self.car.set_data([car_back_x, car_front_x], [car_back_y, car_front_y])
        # wheels
        # plott(x(1)-0.05,sin(3*(x(1)-0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .5 .5]);
        # plot(x(1)+0.05,sin(3*(x(1)+0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .5 .5]);
        # Arrows
        if self.actionArrow is not None:
            self.actionArrow.remove()
            self.actionArrow = None

        if self.actions[a] > 0:
            self.actionArrow = fromAtoB(
                car_front_x, car_front_y,
                car_front_x + self.ARROW_LENGTH *
                np.cos(slope), car_front_y +
                self.ARROW_LENGTH * np.sin(slope),
                #car_front_x + self.CAR_WIDTH*cos(slope)/2., car_front_y + self.CAR_WIDTH*sin(slope)/2.+self.CAR_HEIGHT,
                'k', "arc3,rad=0",
                0, 0, 'simple'
            )
        if self.actions[a] < 0:
            self.actionArrow = fromAtoB(
                car_back_x, car_back_y,
                car_back_x - self.ARROW_LENGTH *
                np.cos(slope), car_back_y -
                self.ARROW_LENGTH * np.sin(slope),
                #car_front_x + self.CAR_WIDTH*cos(slope)/2., car_front_y + self.CAR_WIDTH*sin(slope)/2.+self.CAR_HEIGHT,
                'r', "arc3,rad=0",
                0, 0, 'simple'
            )
        plt.draw()

    def showLearning(self, representation):
        pi = np.zeros(
            (self.X_discretization,
             self.XDot_discretization),
            'uint8')
        V = np.zeros((self.X_discretization, self.XDot_discretization))

        if self.valueFunction_fig is None:
            self.valueFunction_fig = plt.figure("Value Function")
            self.valueFunction_im = plt.imshow(
                V,
                cmap='ValueFunction',
                interpolation='nearest',
                origin='lower',
                vmin=self.MIN_RETURN,
                vmax=self.MAX_RETURN)

            plt.xticks(self.xTicks, self.xTicksLabels, fontsize=12)
            plt.yticks(self.yTicks, self.yTicksLabels, fontsize=12)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\dot x$")

            self.policy_fig = plt.figure("Policy")
            self.policy_im = plt.imshow(
                pi,
                cmap='MountainCarActions',
                interpolation='nearest',
                origin='lower',
                vmin=0,
                vmax=self.actions_num)

            plt.xticks(self.xTicks, self.xTicksLabels, fontsize=12)
            plt.yticks(self.yTicks, self.yTicksLabels, fontsize=12)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\dot x$")
            plt.show()

        for row, xDot in enumerate(np.linspace(self.XDOTMIN, self.XDOTMAX, self.XDot_discretization)):
            for col, x in enumerate(np.linspace(self.XMIN, self.XMAX, self.X_discretization)):
                s = np.array([x, xDot])
                Qs = representation.Qs(s, False)
                As = self.possibleActions()
                pi[row, col] = representation.bestAction(s, False, As)
                V[row, col] = max(Qs)
        self.valueFunction_im.set_data(V)
        self.policy_im.set_data(pi)

        self.valueFunction_fig = plt.figure("Value Function")
        plt.draw()
        self.policy_fig = plt.figure("Policy")
        plt.draw()


#============================================================================================#
#||========================================================================================||#
#||                                      mountain-car                                      ||#
#||========================================================================================||#
#============================================================================================#
class MountainCar(MountainCar_parent):
    actions_num = 3
    state_space_dims = 2
    continuous_dims = [0, 1]

    XMIN = -1.2  # : Lower bound on domain position
    XMAX = 0.6  #: Upper bound on domain position
    XDOTMIN = -0.07  # : Lower bound on car velocity
    XDOTMAX = 0.07  #: Upper bound on car velocity
    INIT_STATE = np.array([-0.5, 0.0])  # : Initial car state
    STEP_REWARD = -1  # : Penalty for each step taken before reaching the goal
    GOAL_REWARD = 0  #: Reward for reach the goal.
    #: X-Position of the goal location (Should be at/near hill peak)
    GOAL = .5
    actions = [-1, 0, 1]
    
    accelerationFactor = 0.001  # : Magnitude of acceleration action
    gravityFactor = -0.0025
    #: Hill peaks are generated as sinusoid; this is freq. of that sinusoid.
    hillPeakFrequency = 3.0

    # Used for visual stuff:
    domain_fig = None
    valueFunction_fig = None
    policy_fig = None
    actionArrow = None
    X_discretization = 20
    XDot_discretization = 20
    CAR_HEIGHT = .2
    CAR_WIDTH = .1
    ARROW_LENGTH = .2

    def __init__(self, start=0.5, noise=0, discount_factor=0.9, episodeCap=1e4):
        self.statespace_limits = np.array(
            [[self.XMIN, self.XMAX], [self.XDOTMIN, self.XDOTMAX]])
        super(MountainCar, self).__init__(start=start, noise=noise, discount_factor=discount_factor, episodeCap=episodeCap)

    def s0(self):
        self.state = self.INIT_STATE.copy()
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def step(self, a):
        """
        Take acceleration action *a*, adding noise as specified in ``__init__()``.
        """
        position, velocity = self.state
        noise = self.accelerationFactor * self.noise * \
            2 * (self.random_state.rand() - .5)
        velocity += (noise +
                     self.actions[a] * self.accelerationFactor +
                     np.cos(self.hillPeakFrequency * position) * self.gravityFactor)
        velocity = bound(velocity, self.XDOTMIN, self.XDOTMAX)
        position += velocity
        position = bound(position, self.XMIN, self.XMAX)
        if position <= self.XMIN and velocity < 0:
            velocity = 0  # Bump into wall
        ns = np.array([position, velocity])
        self.state = ns.copy()
        terminal = self.isTerminal()
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD
        return r, ns, terminal, self.possibleActions()



#============================================================================================#
#||========================================================================================||#
#||                                mountain-car with flags                                 ||#
#||========================================================================================||#
#============================================================================================#
class MountainCar_flag(MountainCar_parent):
    actions_num = 3
    state_space_dims = 2
    continuous_dims = [0, 1]

    XMIN = -1.2  # : Lower bound on domain position
    XMAX = 0.6  #: Upper bound on domain position
    XDOTMIN = -0.07  # : Lower bound on car velocity
    XDOTMAX = 0.07  #: Upper bound on car velocity
    INIT_STATE = np.array([-0.5, 0.0])  # : Initial car state
    STEP_REWARD = -1  # : Penalty for each step taken before reaching the goal
    GOAL_REWARD = 0  #: Reward for reach the goal.
    #: X-Position of the goal location (Should be at/near hill peak)
    GOAL = .5
    actions = [-1, 0, 1]
    
    accelerationFactor = 0.001  # : Magnitude of acceleration action
    gravityFactor = -0.0025
    #: Hill peaks are generated as sinusoid; this is freq. of that sinusoid.
    hillPeakFrequency = 3.0

    # Used for visual stuff:
    domain_fig = None
    valueFunction_fig = None
    policy_fig = None
    actionArrow = None
    X_discretization = 20
    XDot_discretization = 20
    CAR_HEIGHT = .2
    CAR_WIDTH = .1
    ARROW_LENGTH = .2

    def __init__(self, start=0.5, noise=0, discount_factor=1, episodeCap=1e4, 
                 FlagPos=np.array([0]), 
                 FlagWid=np.array([1]), 
                 FlagHeight=np.array([0])):

        self.FlagPos = FlagPos
        self.FlagWid = FlagWid
        self.FlagHeight = FlagHeight
        self.FlagNum = FlagPos.shape[0]
        self.collectedFlags = 0

        self.INIT_STATE = np.hstack(( self.INIT_STATE, [0] ))
        self.statespace_limits = np.array(
            [[self.XMIN, self.XMAX], [self.XDOTMIN, self.XDOTMAX]])
        self.statespace_limits = np.vstack(( self.statespace_limits, [0, self.FlagNum] ))

        super(MountainCar_flag, self).__init__(start=start, noise=noise, 
                                               discount_factor=discount_factor, episodeCap=episodeCap)

    def s0(self):
        self.state = self.INIT_STATE.copy()
        self.collectedFlags = 0
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def step(self, a):
        """ Take acceleration action *a*, adding noise as specified in ``__init__()``. """
        s = self.state
        position, velocity, cflag = self.state
        noise = self.accelerationFactor * self.noise * \
                2 * (self.random_state.rand() - .5)
        velocity += (noise +
                     self.actions[a] * self.accelerationFactor +
                     np.cos(self.hillPeakFrequency * position) * self.gravityFactor)
        velocity = bound(velocity, self.XDOTMIN, self.XDOTMAX)
        position += velocity
        position = bound(position, self.XMIN, self.XMAX)
        if position <= self.XMIN and velocity < 0:
            velocity = 0  # Bump into wall

        r = self.STEP_REWARD
        #------------- calculate the reward and update the last dimension of state -------------#
        ### A. abs(distance)
        # if cflag < self.FlagNum:
        #     dist_s = np.absolute( s[0] - self.FlagPos[int(cflag)] )
        #     dist_ns = np.absolute( position - self.FlagPos[int(cflag)] )
        #     if dist_ns < dist_s:
        #         r += self.FlagHeight[int(cflag)]/dist_s
        #     #---- if cflag is updated ----#
        #     ### (2) the agent is near the flag
        #     if np.absolute(self.FlagPos[int(cflag)] - position) <= 0.1:
        #         cflag += 1
        #         # print('collect flag ', cflag)
        #     ### (1) the agent must pass the flag
        #     # p_low, p_high = min(s[0], position), max(s[0], position)
        #     # if self.FlagPos[int(cflag)] >= p_low and self.FlagPos[int(cflag)] <= p_high:
        #     #     cflag += 1

        ### B. sqr(distance)
        if cflag < self.FlagNum:
            distance_s = s[0] - self.FlagPos[int(cflag)]
            exponent_s = 0.5 * distance_s**2 / self.FlagWid[int(cflag)]
            phi_s = self.FlagHeight[int(cflag)] * np.exp(-exponent_s)
            distance_ns = position - self.FlagPos[int(cflag)]
            exponent_ns = 0.5 * distance_ns**2 / self.FlagWid[int(cflag)]
            phi_ns = self.FlagHeight[int(cflag)] * np.exp(-exponent_ns)
            r += self.discount_factor * phi_ns - phi_s
            #---- if cflag is updated ----#
            p_low, p_high = min(s[0], position), max(s[0], position)
            if self.FlagPos[int(cflag)] >= p_low and self.FlagPos[int(cflag)] <= p_high:
                    cflag += 1
                    # print(cflag)

        ns = np.array([position, velocity, int(cflag)])
        self.collectedFlags = int(cflag)
        self.state = ns.copy()

        terminal = self.isTerminal()
        if terminal: r += self.GOAL_REWARD
        
        return r, ns, terminal, self.possibleActions()



#============================================================================================#
#||========================================================================================||#
#||                       mountain-car with flags at two dimensions                        ||#
#||========================================================================================||#
#============================================================================================#
class MountainCar_flag2(MountainCar_parent):
    actions_num = 3
    state_space_dims = 2
    continuous_dims = [0, 1]

    XMIN = -1.2  # : Lower bound on domain position
    XMAX = 0.6  #: Upper bound on domain position
    XDOTMIN = -0.07  # : Lower bound on car velocity
    XDOTMAX = 0.07  #: Upper bound on car velocity
    INIT_STATE = np.array([-0.5, 0.0])  # : Initial car state
    STEP_REWARD = -1 # -1  # : Penalty for each step taken before reaching the goal
    GOAL_REWARD = 100  #: Reward for reach the goal.
    #: X-Position of the goal location (Should be at/near hill peak)
    GOAL = .5
    actions = [-1, 0, 1]
    
    accelerationFactor = 0.001  # : Magnitude of acceleration action
    gravityFactor = -0.0025
    #: Hill peaks are generated as sinusoid; this is freq. of that sinusoid.
    hillPeakFrequency = 3.0

    # Used for visual stuff:
    domain_fig = None
    valueFunction_fig = None
    policy_fig = None
    actionArrow = None
    X_discretization = 20
    XDot_discretization = 20
    CAR_HEIGHT = .2
    CAR_WIDTH = .1
    ARROW_LENGTH = .2

    def __init__(self, start=0.5, noise=0, discount_factor=1, episodeCap=1e4, 
                 FlagPos=np.array([[0,0]]), 
                 FlagWid=np.array([1]), 
                 FlagHeight=np.array([0])):

        self.FlagPos = FlagPos
        self.FlagWid = FlagWid
        self.FlagHeight = FlagHeight
        self.FlagNum = FlagPos.shape[0]
        self.collectedFlags = 0

        self.INIT_STATE = np.hstack(( self.INIT_STATE, [0] ))
        self.statespace_limits = np.array(
            [[self.XMIN, self.XMAX], [self.XDOTMIN, self.XDOTMAX]])
        self.statespace_limits = np.vstack(( self.statespace_limits, [0, self.FlagNum] ))

        super(MountainCar_flag2, self).__init__(start=start, noise=noise, 
                                               discount_factor=discount_factor, episodeCap=episodeCap)

    def s0(self):
        self.state = self.INIT_STATE.copy()
        self.collectedFlags = 0
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def step(self, a):
        """ Take acceleration action *a*, adding noise as specified in ``__init__()``. """
        s = self.state
        position, velocity, cflag = self.state
        noise = self.accelerationFactor * self.noise * \
                2 * (self.random_state.rand() - .5)
        velocity += (noise +
                     self.actions[a] * self.accelerationFactor +
                     np.cos(self.hillPeakFrequency * position) * self.gravityFactor)
        velocity = bound(velocity, self.XDOTMIN, self.XDOTMAX)
        position += velocity
        position = bound(position, self.XMIN, self.XMAX)
        if position <= self.XMIN and velocity < 0:
            velocity = 0  # Bump into wall

        #------------- calculate the reward and update the last dimension of state -------------#
        r = self.STEP_REWARD
        if cflag < self.FlagNum:
            exponent_s = np.divide(np.sum(0.5*(s[0:-1]-self.FlagPos[int(cflag)])**2), self.FlagWid[int(cflag)])
            phi_s = np.multiply(self.FlagHeight[int(cflag)], np.exp(-exponent_s))
            ns_pv = np.array([position, velocity])
            exponent_ns = np.divide(np.sum(0.5*(ns_pv-self.FlagPos[int(cflag)])**2), self.FlagWid[int(cflag)])
            phi_ns = np.multiply(self.FlagHeight[int(cflag)], np.exp(-exponent_ns))
            r += self.discount_factor * phi_ns - phi_s
            p_low, p_high = min(s[0], position), max(s[0], position)
            v_low, v_high = min(s[1], velocity), max(s[1], velocity)
            #---- if cflag is updated ----#
            # (1) the agent must pass the flag
            # if np.all(self.FlagPos[int(cflag)] >= np.array([p_low, v_low])) and \
            #     np.all(self.FlagPos[int(cflag)] <= np.array([p_high, v_high])):
            #     cflag += 1
            # (2) the agent is near the flag
            if np.absolute(position - self.FlagPos[int(cflag),0]) <= 0.1 and \
                np.absolute(velocity - self.FlagPos[int(cflag),1]) <= 0.01:
                cflag += 1

        ns = np.array([position, velocity, int(cflag)])
        self.collectedFlags = int(cflag)
        self.state = ns.copy()

        terminal = self.isTerminal()
        if terminal: r += self.GOAL_REWARD
        
        return r, ns, terminal, self.possibleActions()