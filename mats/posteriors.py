import numpy as np
import scipy as sp
import scipy.stats

class Posterior():

    @property
    def mean(self):
        raise NotImplementedError()

    def update(self, x):
        raise NotImplementedError()


class BetaPosterior(Posterior):
    
    def __init__(self, alpha=0.5, beta=0.5):
        self.a = alpha
        self.b = beta

    @property
    def mean(self):
        return self.a / (self.a + self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x
    
    def sample(self):
        return sp.stats.beta(a=self.a, b=self.b).rvs(1)[0]


################

class WindFarmGaussianPosterior(Posterior):

    def __init__(self, std, action, power_curve, type='jeffreys'):
        self._mu0 = power_curve[action]
        self._std0 = 1000000 if action != 0.0 else 0


        self._std = std
        self._type = type
        self._count = 0
        self._sum = 0
        self._sumsq = 0

        # Check whether prior type exists
        if type == 'jeffreys':
            # Improper mean
            self._mu = np.nan
        else:
            raise ValueError('This prior type does not exist')
                
    @property
    def mean(self):
        return self._sum / self._count

    @property
    def _sigma(self):
        return self._std / np.sqrt(self._count)

    def update(self, x):
        if self._count == 0:
            self._sum = x
            self._count += 1
        elif self._count == 1:
            old_x = self._sum
            self._sum += x
            self._count += 1
            self._sumsq = (old_x - self.mean)**2 + (x - self.mean)**2
        else:
            old_mean = self.mean
            self._sum += x
            self._count += 1
            self._sumsq += (x - old_mean) * (x - self.mean)

    def sample(self):
        if self._count < 1 and self._std0 > 0.0:
            x = sp.stats.norm(loc=self._mu0, scale=self._std0).rvs(1)[0]
            return np.min([x, 8000000])
        elif self._std0 == 0:
            return self._mu0
        else:
            return self.mean

    def _sample_gaussian(self):
        return sp.stats.norm(loc=self.mean, scale=self._sigma).rvs(1)[0]

    def _sample_uniform(self):
        return sp.stats.uniform(loc=0, scale=8*10**6).rvs(1)[0]

    def _sample_student_t(self):
        scale = np.sqrt(self._sumsq / self._count / self._count)
        if scale == 0:
            return np.nan
        else:
            return sp.stats.t(df=self._count, loc=self.mean, scale=scale).rvs(1)[0]
