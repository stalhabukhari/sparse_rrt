from sparse_rrt import _sst_module
import numpy as np


class WithEuclideanDistanceComputer(object):
    '''
    Add euclidian distance computer to a cpp system class
    '''
    def distance_computer(self):
        return _sst_module.euclidean_distance(np.array(self.is_circular_topology()))


class Car(_sst_module.Car, WithEuclideanDistanceComputer):
    pass


class CartPole(_sst_module.CartPole, WithEuclideanDistanceComputer):
    pass


class Pendulum(_sst_module.Pendulum, WithEuclideanDistanceComputer):
    pass


class Point(_sst_module.Point, WithEuclideanDistanceComputer):
    pass


class RallyCar(_sst_module.RallyCar, WithEuclideanDistanceComputer):
    pass


class TwoLinkAcrobot(_sst_module.TwoLinkAcrobot):
    '''
    Acrobot has its own custom distance for faster convergence
    '''
    def distance_computer(self):
        return _sst_module.TwoLinkAcrobotDistance()


class RectangleObs3D(_sst_module.RectangleObs3DSystem):
    def __init__(self, obstacle_list, obstacle_width, env_name):
        super().__init__(obstacle_list, obstacle_width, env_name)
        self.env_name = env_name
    def distance_computer(self):
       if self.env_name == 'quadrotor':
            return _sst_module.QuadrotorDistance()
