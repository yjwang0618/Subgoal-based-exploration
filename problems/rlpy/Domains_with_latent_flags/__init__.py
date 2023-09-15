from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
#from Domain import Domain
from future import standard_library
standard_library.install_aliases()

from .GridWorld import GridWorld, GridWorld_Flag
from .GridWorld_Items import GridWorld_Items, GridWorld_Items_Flag
from .GridWorld_PuddleWindy import GridWorld_PuddleWindy, GridWorld_PuddleWindy_Flag

from .MountainCar import MountainCar, MountainCar_flag

from .PuddleWorld import PuddleWorld