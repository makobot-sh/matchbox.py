import math
from Messages import Messages as msg
from Gaussian import Gaussian
from Gaussian import PointMass
from Gaussian import Uniform
import sys
from collections import defaultdict
import copy
from tqdm import tqdm
import pickle
from datetime import datetime
import random 

assert sys.version_info >= (3, 9), "Use Python 3.9 or newer"

class symmbreakdefaultdict(defaultdict):
    traitCount: int = None
    def __missing__(self, key):
        ret = self[key] = self.defaultItemPrior()
        return ret
        
    def defaultItemPrior(self):
        assert (self.traitCount is not None), "traitCount attribute must be set before assigning items"
        size = len(self)
        if size < self.traitCount:
            for prev in self:
                self[prev][size] = PointMass(0)
            return [PointMass(1) if i==size else PointMass(0) if i<size else Gaussian(0,1) for i in range(self.traitCount)]
        else:
            return [Gaussian(0,1) for _ in range(self.traitCount)]

class RatingEvent:
    """Stores likelihood information for a rating event.

    Attributes:
    r -- user, movie and value of the rating
    U8 -- user traits likelihood after observing the rating
    V8 -- movie traits likelihood after observing the rating
    ubias8 -- user bias likelihood after observing the rating
    vbias8 -- movie bias likelihood after observing the rating
    """

    r: tuple[int,int,float]
    U8: dict[int, list[Gaussian]]
    V8: dict[int, list[Gaussian]]
    ubias8: dict[int, Gaussian]
    vbias8: dict[int, Gaussian]
    def __init__(self, r:tuple[int,int,float], 
                 U8: dict[int, list[Gaussian]],    
                 V8: dict[int, list[Gaussian]],
                 ubias8: dict[int, Gaussian],
                 vbias8: dict[int, Gaussian],
                 U_thr8: dict[int, list[Gaussian]]):
        self.r = r
        self.U8 = U8
        self.V8 = V8
        self.ubias8 = ubias8
        self.vbias8 = vbias8
        self.U_thr8 = U_thr8

class Matchbox:
    """ Matchbox recommender system.

    Attributes:
    traitCount -- number of traits to model.
    numThresholds -- number of user rating thresholds to model.
    U -- user latent trait space.
    V -- item latent trait space.
    U_thr -- user latent rating threshold space.
    ubias -- user latent bias space.
    vbias -- item latent bias space.
    betaNoise2 -- rating noise variance.
    biasNoise2 -- bias noise variance.
    tauNoise2 -- user threshold noise variance.
    ratingHistory -- array containing all ratings the recommender has observed in the order they were added.
    logEvidence -- sum of logarithms of first predictions ('s probabilities) for observed ratings. Not updated during convergence.
    logEvidenceHistory -- array containing log of predictions ('s probabilities) for observed ratings, updated during convergence.
    """
    traitCount: int
    numThresholds: int
    U: defaultdict[int,list[Gaussian]]
    V: symmbreakdefaultdict[int,list[Gaussian]]
    U_thr: defaultdict[int, list[Gaussian]]
    ubias: defaultdict[int,Gaussian]
    vbias: defaultdict[int,Gaussian]
    betaNoise2: float
    biasNoise2: float
    tauNoise2: float
    ratingHistory: list[RatingEvent]
    logEvidence: float
    logEvidenceHistory: list[float]

    def __init__(self, traitCount=2, numThresholds=1, betaNoise2=1, biasNoise2=0, tauNoise2=0):
        self._traitCount = traitCount
        self.numThresholds = numThresholds
        self.betaNoise2 = betaNoise2
        self.biasNoise2 = biasNoise2
        self.tauNoise2 = tauNoise2        
        self.U = defaultdict(self._initUDefaultDict)
        self.U_thr = defaultdict(self._initUthrDefaultDict)
        self.V = symmbreakdefaultdict()
        self.V.traitCount = traitCount
        self.ubias = defaultdict(self._initBiasDefaultDict)
        self.vbias = defaultdict(self._initBiasDefaultDict)
        self.logEvidence = 0
        self.ratingHistory = []
        self.logEvidenceHistory = []

    def _initUDefaultDict(self):
        return [Gaussian(0,1) for _ in range(self.traitCount)]
    
    def _initUthrDefaultDict(self):
        return msg.thresholdPriors(self.numThresholds)
    
    def _initBiasDefaultDict(self):
        return Gaussian(0,1)
    
    def __setstate__(self, d):
        self.__dict__ = d
        self.traitCount = d["_traitCount"] #Important to trigger setter

    def save(self, picklePath):
        with open(picklePath,"wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(picklePath):
        with open(picklePath, "rb") as f:
            r = pickle.load(f)
        return r

    @property
    def traitCount(self):
        return self._traitCount
    
    @traitCount.setter
    def traitCount(self, value):
        self._traitCount = value
        self.V.traitCount = value

    def addRating(self, userId: int, movieId: int, rating: float) -> None:
        """Add a new rating to the system."""
        assert rating >= 0 and rating <= self.numThresholds, f"Rating of {rating} is outside of bounds of rating system"
        likelihoods_and_pred = self._rateAndUpdatePriors(userId, movieId, rating)
        self.ratingHistory.append(RatingEvent((userId, movieId, rating), *likelihoods_and_pred[:-1]))
        pred = likelihoods_and_pred[-1]
        self.logEvidence += pred
        self.logEvidenceHistory.append(pred)

    def convergeModel(self, picklePath=None, it_start=0, i_start=0) -> None:
        """Propagate rating information through the model by re-observing all ratings in the rating history until all latent trait marginals converge."""
        if len(self.ratingHistory) < 2:
            return
        print(f"Convergiendo evento {len(self.ratingHistory)-1}")
        convergence = False
        it = it_start
        ubias_old = copy.deepcopy(self.ubias)
        while (not (convergence or it>=5)):
            print(f"========== History it. {it} ==========")
            if it > it_start:
                i_st = 0
            else:
                i_st = i_start
            for i, evt in enumerate(tqdm(self.ratingHistory,initial=i_st)):
                userId = evt.r[0]
                self.unviewRating(evt)
                likelihoods_and_pred = self._rateAndUpdatePriors(*evt.r)
                self.ratingHistory[i] = RatingEvent(evt.r, *likelihoods_and_pred[:-1])
                pred = likelihoods_and_pred[-1]
                self.logEvidenceHistory[i] = pred
                #convergence = all(msg.estimationsConverge(uki,nuki) for uk, nuk in zip(ubias_old.values(), self.ubias.values()) for uki, nuki in zip(uk, nuk))
                convergence = all(msg.estimationsConverge(uki,nuki) for uki, nuki in zip(ubias_old.values(), self.ubias.values()))
                #print("\n".join([f"{k}: {ubias_old[k]} || {kn}: {self.ubias[kn]}" for k,kn in zip(ubias_old.keys(), self.ubias.keys())]))
                ubias_old[userId] = self.ubias[userId] #copy updated ubias to check convergence in next step
                if picklePath is not None and i%1000==0:
                    self.save(f"{picklePath}_it{it}_i{i}.p")
            it += 1
            if picklePath is not None:
                self.save(f"{picklePath}_it{it}_i{i}.p")
            print(f"| Ev geo mean  |{f'{self.geoMeanEvidence():.5f}':^11}|{f'{self.geoMeanConvergedEvidence():.5f}':^11}|")

    def unviewRating(self, e: RatingEvent) -> None:
        """Updates trait marginals to what they would be if the rating event had not been observed."""
        user = e.r[0]
        item = e.r[1]
        self.dictDiv(self.U, e.U8, user)
        self.dictDiv(self.V, e.V8, item)
        self.dictDiv(self.ubias, e.ubias8, user)
        self.dictDiv(self.vbias, e.vbias8, item)
        self.dictDiv(self.U_thr, e.U_thr8, user)

    @staticmethod
    def dictDiv(source: dict[int, any], div: RatingEvent, k: int):
        if isinstance(source[k], list): # for nested [int, list[Gaussian]] dict
            source[k] = [sk/dk for sk,dk in zip(source[k],div)]
        else: # for [int, Gaussian] dict
            source[k] = source[k]/div

    def _rateAndUpdatePriors(self, userId: int, movieId: int, rating: float) -> None:
        userPrior = self.U[userId]
        moviePrior = self.V[movieId]
        userBiasPrior = self.ubias[userId]
        movieBiasPrior = self.vbias[movieId]
        thrPrior = self.U_thr[userId]
        biasPrior = msg.one_bias(userBiasPrior, movieBiasPrior)
        userLhood, movieLhood, biasLhood, thrLhood, pred = self._rate(userPrior, moviePrior, biasPrior, thrPrior, rating)
        self.U[userId] = [margUki * m8uki for margUki, m8uki in zip(userPrior, userLhood)] #New prior (uki marginal probability, prior * msg8)
        self.V[movieId] = [margVki * m8vki for margVki, m8vki in zip(moviePrior, movieLhood)] #idem uki
        userBiasLhood = msg.lhood_subjectBias(biasLhood, self.vbias[movieId])
        movieBiasLhood = msg.lhood_subjectBias(biasLhood, self.ubias[userId])
        # Posterior = lhood*prior
        self.ubias[userId] = userBiasPrior * userBiasLhood
        self.vbias[movieId] = movieBiasPrior * movieBiasLhood
        self.U_thr[userId] = [prior * lhood for prior, lhood in zip(thrPrior, thrLhood)]
        return (userLhood, movieLhood, userBiasLhood, movieBiasLhood, thrLhood, pred)

    def _rate(self, 
              sk: list[Gaussian], 
              tk: list[Gaussian], 
              b: Gaussian, 
              user_thr: list[Gaussian], 
              rObs: float
              ) -> tuple[list[Gaussian], list[Gaussian], list[Gaussian], Gaussian]:
        convergence = False
        it = 0
        firstPred = 0
        pr_old = math.inf
        m7S = [Uniform() for _ in range(self.traitCount)] #dont have necessary data yet to compute msg7, so we initialize to Uniform
        m7T = [Uniform() for _ in range(self.traitCount)] #same as above
        b1 = b + Gaussian(0, self.biasNoise2)
        priorThr = user_thr

        while (not (convergence or it >= 100)):
            sk1 = [ski * m7si for ski, m7si in zip(sk, m7S)]
            tk1 = [tki * m7ti for tki, m7ti in zip(tk, m7T)]
            m2 = msg.two(sk1, tk1)
            newRating = msg.four(b1, m2, self.betaNoise2)
            m5, thrLhood, prs = msg.five(newRating, rObs, priorThr, self.tauNoise2, plot=False)
            m5p = Gaussian(m5.mu, math.sqrt(m5.sigma2+self.betaNoise2))
            m6 = msg.six(m5p, b1, m2)
            m7S = msg.seven(m6, tk1) #m7 is posterior in collaborative filtering, but truthfully the posterior is msg8. msg7 is missing substracting other "active" columns like substracting vb for ub bias posterior
            m7T = msg.seven(m6, sk1)
            
            # Chequeo convergencia
            if it==0:
                firstPreds = prs
                #print(f"{sum(firstPreds)}: {firstPreds}")
                firstPred = math.log(prs[rObs])
            convergence = abs(math.log(prs[rObs]) - pr_old)<0.00000001; pr_old = math.log(prs[rObs])
            it += 1
        #print(f"sum: {sum(firstPreds)}, [{', '.join(f'{q:.5f}' for q in firstPreds)}]")
        posteriorb = msg.seven_bias(m5p, m2, self.biasNoise2)
        return (m7S, m7T, posteriorb, thrLhood, firstPred)     

    def geoMeanEvidence(self) -> float:
        """Evidence made of predictions when ratings were first observed."""
        return math.exp(self.logEvidence/len(self.logEvidenceHistory))
    
    def geoMeanConvergedEvidence(self) -> float:
        """Evidence made of predictions when ratings were last observed (aka evidence of observations after convergence)."""
        return math.exp(sum(self.logEvidenceHistory)/len(self.logEvidenceHistory))

    def printEvidence(self):
        """Print evidence and geometric mean, both of first observations and updated with convergence."""
        print()
        print(f"|{' ':^11}| add, 1st  | add, conv |")
        print(f"|{'|'.join('-'*11 for _ in range(3))}|")
        print(f"| log(evid) |{f'{self.logEvidence:.5f}':^11}|{f'{sum(self.logEvidenceHistory):.5f}':^11}|")
        print(f"| geo mean  |{f'{self.geoMeanEvidence():.5f}':^11}|{f'{self.geoMeanConvergedEvidence():.5f}':^11}|")
        print()

    def printConfigs(self):
        configs = {"traitCount": self.traitCount,
        "numThresholds": self.numThresholds,
        "betaNoise2": self.betaNoise2,
        "biasNoise2": self.biasNoise2,
        "tauNoise2": self.tauNoise2
        }
        print(configs)

    def printLatent(self, numUsers=math.inf, numItems=math.inf):
        """Print latent user and item variables (traits, bias, threhsolds)."""
        print(f'| movieId |{"|".join([f"{x:^28}" for x in [f"trait_{i}" for i in range(self.traitCount)]+["bias"]])}|')
        print(f"|---------|{'|'.join('-'*28 for _ in range(self.traitCount+1))}|")
        i = 0
        for k, v in self.V.items():
            if i==numUsers:
                break
            print(f'|{k:^9}|{ "|".join([f"{x:^28}" for x in v+[self.vbias[k]]]) }|')
            i+=1

        print()
        print(f'| userId  |{"|".join([f"{x:^28}" for x in [f"trait_{i}" for i in range(self.traitCount)]+["bias"]])}|')
        print(f"|---------|{'|'.join('-'*28 for _ in range(self.traitCount+1))}|")
        i = 0
        for k, v in self.U.items():
            if i==numItems:
                break
            print(f'|{k:^9}|{ "|".join([f"{x:^28}" for x in v+[recommender.ubias[k]]]) }|')
            i+=1

        print()
        print(f'| userId  |{"|".join([f"{x:^28}" for x in [f"thr_{i}" for i in range(self.numThresholds)]])}|')
        print(f"|---------|{'|'.join('-'*28 for _ in range(self.numThresholds))}|")
        i = 0
        for k, v in self.U.items():
            if i==numItems:
                break
            print(f'|{k:^9}|{ "|".join([f"{x:^28}" for x in self.U_thr[k]]) }|')
            i+=1

if __name__ == "__main__":
    # Some example code
    recommender = Matchbox()
    recommender.traitCount = 4
    recommender.numThresholds = 4
    recommender.printConfigs()
    now = datetime.now().strftime("%d%m%Y_%H-%M")
    print("Training...")
    with open("data/MovieLens_binary_100k/ratings_train.csv", "r") as f:
        for line in tqdm(f.readlines(),initial=0,total=75000):
            l = line.split(",")
            recommender.addRating(int(l[0]),int(l[1]),int(l[2]))
    recommender.printConfigs()
    recommender.printEvidence()
    recommender.convergeModel()
    recommender.printEvidence()

    print("Testing...")
    with open("data/MovieLens_binary_100k/ratings_test.csv", "r") as f:
        for line in tqdm(f.readlines(),initial=0,total=25000):
            l = line.split(",")
            recommender.addRating(int(l[0]),int(l[1]),int(l[2]))
    recommender.printConfigs()
    recommender.printEvidence()

    print("Converging...")
    recommender.convergeModel()
    print("After convergence")
    recommender.printConfigs()
    recommender.printEvidence()
