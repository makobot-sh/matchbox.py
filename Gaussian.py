from scipy.stats import norm 
import math
from scipy.stats import truncnorm

class Gaussian:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2
        # Non-centered second moment
        self.ncsm = self.mu**2+self.sigma2

    def eval(self, x: [float | list[float]]) -> [float | list[float]]:
        return norm.pdf(x, self.mu, self.sigma)

    def sample(self, size=1):
        return norm.rvs(self.mu, self.sigma, size=size)

    def trunc(self, a: float, b: float):
        mean = self.mu  # Mean of the original Gaussian distribution
        std_dev = self.sigma  # Standard deviation of the original Gaussian distribution
        # Create a truncated normal distribution object
        trunc_norm = truncnorm((a - mean) / std_dev, (b - mean) / std_dev, loc=mean, scale=std_dev)
        return( Gaussian(trunc_norm.mean(),trunc_norm.std()) )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == math.inf: 
                return Uniform()
            else:
                return Gaussian(other*self.mu, abs(other)*self.sigma)
        else:
            if self.sigma == 0.0 or other.sigma == 0.0:
                mu = self.mu/((self.sigma**2/other.sigma**2) + 1) if self.sigma == 0.0 else other.mu/((other.sigma**2/self.sigma**2) + 1)
                sigma = 0.0
            else:
                _tau, _pi = self.tau + other.tau, self.pi + other.pi
                mu, sigma = Gaussian.mu_sigma(_tau, _pi)
            
            if sigma == math.inf:
                return Uniform()
            elif sigma == 0:
                return PointMass(mu)
            return Gaussian(mu, sigma)
        # OLD
        if (other == 0):
            return PointMass(0)
        if (other == 1):
            return self
        elif (isinstance(other, (int, float))):
            # source: https://math.stackexchange.com/questions/275648/multiplication-of-a-random-variable-with-constant
            return Gaussian(self.mu*other, self.sigma*other)
        elif (isinstance(other, PointMass)):
            return PointMass(self.mu*other.mu)
        elif (isinstance(other, Uniform)):
            return self
        elif (isinstance(other, Gaussian)):
            return Gaussian.mul(self, other)
        else:
            raise NotImplementedError(f"Multiplying Gaussian by element of type {type(other)} is unsupported. Value of other: {other}")
    def __rmul__(self, other):
        return self.__mul__(other)
    def __matmul__(self, other):
        return self.__mul__(other)
    def __rmatmul__(self, other):
        return self.__rmul__(other)
    def __truediv__(self, other):
        if (other == 0):
            raise ZeroDivisionError(f"Tried to divide a Gaussian by zero")
        if (other == 1):
            return self
        elif (isinstance(other, (int, float))):
            return Gaussian(self.mu/other, self.sigma/other)
        elif (isinstance(other, PointMass)):
            assert self.sigma2 != 0, f"This might be a pointmass: {self}"
            raise ZeroDivisionError()
            #raise NotImplementedError(f"Dividing Gaussian by element of type {type(other)} is unsupported. Value of other: {other}")
            # De donde salio esto: ???
            #return PointMass(other.mu/((other.sigma**2/self.sigma**2) + 1))
        elif (isinstance(other, Uniform)):
            return self
        elif (isinstance(other, Gaussian)):
            return Gaussian.div(self, other)
        else:
            raise NotImplementedError(f"Dividing Gaussian by element of type {type(other)} is unsupported. Value of other: {other}")
    def __add__(self, other):
        if (other == 0):
            return self
        elif (isinstance(other, Gaussian)):
            return Gaussian(self.mu+other.mu, math.sqrt(self.sigma2+other.sigma2))
        else:
            raise NotImplementedError(f"Adding Gaussian with element of type {type(other)} is unsupported. Value of other: {other}")
    def __radd__(self, other):
        if (other == 0):
            return self
        elif (isinstance(other, Gaussian)):
            return Gaussian(self.mu+other.mu, math.sqrt(self.sigma2+other.sigma2))
        else:
            raise NotImplementedError(f"Adding Gaussian with element of type {type(other)} is unsupported. Value of other: {other}")
    def __sub__(self, other):
        if (other == 0):
            return self
        elif (isinstance(other, Gaussian)):
            return Gaussian(self.mu-other.mu, math.sqrt(self.sigma2+other.sigma2))
        else:
            raise NotImplementedError(f"Substracting element of type {type(other)} from Gaussian is unsupported. Value of other: {other}")
    def __repr__(self):
        return f"N(mu={self.mu:.4f}, var={self.sigma2:.4f})"
    def __format__(self, format_spec):
        return self.__repr__().__format__(format_spec)
    
    @staticmethod
    def mul(norm1, norm2):
        sigma2Star = (1/norm1.sigma2 + 1/norm2.sigma2)**(-1) #c*N(mu, sigma) = N(c*mu, c*sigma) #TODO: agregar al pdf
        muStar = norm1.mu/norm1.sigma2 + norm2.mu/norm2.sigma2
        muStar *= sigma2Star
        sigma = math.sqrt(sigma2Star)
        if sigma==math.inf:
            return Uniform()
        elif sigma==0:
            return PointMass(muStar)
        else:
            return Gaussian(muStar, sigma) 
        #c = Gaussian(norm2.mu, math.sqrt(norm1.sigma2+norm2.sigma2)).eval(norm1.mu) #result should be multiplied by c, but is proportional to not multiplying by it
        return Gaussian(muStar, math.sqrt(sigma2Star))

    @staticmethod
    def div(norm1, norm2):
        _tau = norm1.tau - norm2.tau; _pi = norm1.pi - norm2.pi
        mu, sigma = Gaussian.mu_sigma(_tau, _pi)
        if sigma==math.inf:
            return Uniform()
        elif sigma==0:
            return PointMass(mu)
        else:
            return Gaussian(mu, sigma) 
        # OLD
        sigma2Star = (1/norm1.sigma2 - 1/norm2.sigma2)**(-1) #c*N(mu, sigma) = N(c*mu, c*sigma) #TODO: agregar al pdf
        muStar = norm1.mu/norm1.sigma2 - norm2.mu/norm2.sigma2
        muStar *= sigma2Star
        #c = Gaussian(norm2.mu, math.sqrt(norm1.sigma2+norm2.sigma2)).eval(norm1.mu) #result should be multiplied by c, but is proportional to not multiplying by it
        return Gaussian(muStar, math.sqrt(sigma2Star))

    @staticmethod
    def mu_sigma(tau_,pi_):
        if pi_ > 0.0:
            sigma = math.sqrt(1/pi_)
            mu = tau_ / pi_
        elif pi_ + 1e-5 < 0.0:
            raise ValueError(" sigma should be greater than 0 ")
        else:
            sigma = math.inf 
            mu = 0.0
        return mu, sigma

    @property
    def tau(self):
        if self.sigma > 0.0:
            return self.mu * (self.sigma**-2)
        else:
            return math.inf
        
    @property
    def pi(self):
        if self.sigma > 0.0:
            return self.sigma**-2
        else:
            return math.inf

class Uniform(Gaussian):
    def __init__(self):
        super(Uniform, self).__init__(0, math.inf)
    def __truediv__(self, other):
        raise NotImplementedError("Can't divide a Uniform (inf uncertainty) by anything bc denominator must have bigger uncertainty") #TODO: ahcer esto bien
    def __repr__(self):
        return f"Uniform()"
    
class PointMass(Gaussian):
    def __init__(self, mu):
        super(PointMass, self).__init__(mu, 0)
    def __mul__(self, other):
        if (other == 0):
            return PointMass(0)
        elif (other == 1):
            return self
        elif (isinstance(other, (int, float))):
            # source: https://math.stackexchange.com/questions/275648/multiplication-of-a-random-variable-with-constant
            return PointMass(self.mu*other)
        elif (isinstance(other, PointMass)):
            return PointMass(self.mu*other.mu)
        elif (isinstance(other, Uniform)):
            return self
        elif (isinstance(other, Gaussian)):
            return PointMass(self.mu/((self.sigma**2/other.sigma**2) + 1))
        else:
            raise NotImplementedError(f"Adding PointMass with element of type {type(other)} is unsupported. Value of other: {other}")
    def __truediv__(self, other):
        if (other == 0):
            raise ZeroDivisionError(f"Tried to divide a PointMass by zero")
        if (other == 1):
            return self
        elif (isinstance(other, (int, float))):
            return PointMass(self.mu/other)
        if (isinstance(other, PointMass)):
            if self.mu != other.mu:
                raise ZeroDivisionError()
            return Uniform()
        elif (isinstance(other, Gaussian)):
            return self
        else:
             raise NotImplementedError(f"Dividing PointMass by element of type {type(other)} is unsupported. Value of other: {other}")
    def __repr__(self):
        return f"PointMass({self.mu:.4f})"