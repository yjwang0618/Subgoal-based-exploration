from abc import ABCMeta, abstractproperty

from miso_IS_gw_ import misoGridWorld_xs


class MisoGridWorld10(object):
    
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, version, obj_func_idx, bucket):
        self.replication_no = replication_no
        self.version = version
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func = [misoGridWorld_xs(version, mult=1.0)]
        self._obj_func_idx = obj_func_idx

    @property
    def obj_func_min(self): return self._obj_func[self._obj_func_idx]

    @property
    def num_iterations(self): return 100

    @property
    def hist_data(self): return self._hist_data

    @hist_data.setter
    def set_hist_data(self, data): self._hist_data = data

    # comment the following when there is no multi information source
    @abstractproperty
    def num_is_in(self): None

    @property
    def truth_is(self): return 0

    @property
    def exploitation_is(self): return 1

    @property
    def list_sample_is(self): return range(4)



class MisoGridWorld10Mkg(MisoGridWorld10):

    def __init__(self, replication_no, version, obj_func_idx, bucket):
        super(MisoGridWorld10Mkg, self).__init__(replication_no, "mkg", version, obj_func_idx, bucket)

    @property
    def num_is_in(self): return 3   # This should be idx of the last IS


class_collection = {
    "miso_ISgw10": MisoGridWorld10Mkg,
}
