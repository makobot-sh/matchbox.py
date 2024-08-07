from Gaussian import Gaussian
from Gaussian import PointMass
from Gaussian import Uniform
from scipy.stats import norm 
import math
#import numpy as np
from scipy.special import rel_entr
import matplotlib.pyplot as plt

EPSILON = 1e-7
PLT_COUNT = 1

class Messages:
    @staticmethod
    def thresholdPriors(numThresholds):
        ##if numThresholds == 1:
        ##    res.append([PointMass(0)])
        ##    continue
        thresholds = [Gaussian(l - numThresholds / 2.0 + 0.5, 1.0) for l in range(numThresholds)]
        #if numThresholds%2 == 1 and numThresholds>1:
        #    thresholds[int((numThresholds+1)/2)-1] = PointMass(0) #Matchbox seems to fix side and middle thresholds in -inf, 0, +inf
        assert(all(thresholds[i].mu > thresholds[i-1].mu+0.1 for i in range(1, len(thresholds)-1)))
        return thresholds

    @staticmethod
    def one_bias(ubias: Gaussian, vbias: Gaussian) -> Gaussian:
        return ubias + vbias

    @staticmethod
    def one(U: list[list[Gaussian]], x: list[Gaussian]) -> list[Gaussian]:
        return [uki*xi for uk in U for uki, xi in zip(uk, x)]

    @staticmethod
    def two(margS: list[Gaussian], margT: list[Gaussian]) -> list[Gaussian]:
        """Message 2. Descending message from multiplication factor."""
        return [Gaussian(sk.mu*tk.mu, math.sqrt(sk.sigma2*tk.sigma2 + sk.sigma2*tk.mu**2 + tk.sigma2*sk.mu**2)) for sk, tk in zip(margS, margT)]

    @staticmethod
    def seven_bias(m5p: Gaussian, normZ: list[Gaussian], biasNoise2: float) -> Gaussian:
        """Update message sent to b variable after observing rating."""
        return m5p-(sum(normZ))-Gaussian(0, biasNoise2)

    @staticmethod
    def lhood_subjectBias(normB: Gaussian, normVbias: Gaussian) -> Gaussian:
        """Update message sent to ubias and vbias variables after observing rating."""
        return normB-normVbias

    @staticmethod
    def four(b: Gaussian, normZ: list[Gaussian], betaNoise2: float) -> Gaussian:
        """Message 4. (Continuous) Rating estimate."""
        sumMuZ = sum([zk.mu for zk in normZ])
        sumSigma2Z = sum([zk.sigma2 for zk in normZ])
        return Gaussian(b.mu + sumMuZ, math.sqrt(betaNoise2 + b.sigma2 + sumSigma2Z))

    @staticmethod
    def five(norm4: Gaussian, rObs: int, priorThr: list[Gaussian], thrNoise2: float, plot:bool, predict:bool = False) -> Gaussian:
        if len(priorThr) == 1:
            m9 = norm4
            d = m9 - priorThr[0]
            m10 = d.trunc(a=0,b=math.inf)/d if rObs==1 else d.trunc(a=-math.inf,b=0)/d
            m12 = norm4 - m10
            evidencias = Messages.evidences([d])
            return (m10+priorThr[0], [m12], evidencias)

        L = len(priorThr)
        evidencia = 0
        convergencia_umbrales = False; old_mu_d0 = - math.inf; old_sigma_d0 = math.inf
        i_umbrales = 0
        Nds = [Gaussian(0,math.inf) for _ in priorThr]
        old_ds = Nds.copy()
        N10 = [Gaussian(0,math.inf) for _ in range(L)]
        N11 = [Gaussian(0,math.inf) for _ in range(L)]
        N12 = [Gaussian(0,math.inf) for _ in range(L)]

        while not (convergencia_umbrales or i_umbrales >= 20):
            for i_thr, thr in enumerate(priorThr):
                N9 = norm4 * math.prod(N11[:i_thr]) * math.prod(N11[i_thr+1:])
                #N9 = [norm4 * np.prod([N11[l_other] for l_other in range(L) if l != l_other]) for l in range(L)]
                Nd = N9 - thr
                Nds[i_thr] = Nd
                #Nds = [N9[l] - priorThr[l] for l in range(L)] # Prior P(di|bi,r), mensaje 9 prima
                N10[i_thr] = Nd.trunc(a=-math.inf,b=0)/Nd if i_thr >= rObs else Nd.trunc(a=0,b=math.inf)/Nd
                #N10 = [Nds[l].trunc(a=-math.inf,b=0)/Nds[l] if l >= rObs else Nds[l].trunc(a=0,b=math.inf)/Nds[l] for l in range(L)]
                N11[i_thr] = thr + N10[i_thr]
                N12[i_thr] = N9-N10[i_thr]
                #print(diff, i_umbrales)
            diffs = [max(abs(Nds[i].mu - old_ds[i].mu), abs(Nds[i].sigma - old_ds[i].sigma)) for i in range(len(priorThr))]
            convergencia_umbrales = all([diff < 0.001 for diff in diffs])
            convergencia_umbrales = True
            old_ds = Nds.copy()
            if i_umbrales==0 :
                # multiplicación de probabilidad de predicción correcta para cada umbral
                evidencias = Messages.evidences(Nds)
                if False and plot:
                    tots = Messages.evidencePermutations(Nds)
                    print("Probas for all possible overcome thr permutations:")
                    print(f"sum: {sum(tots)}, [{', '.join(f'{q:.5f}' for q in tots)}]")
                    pass
                    #Messages.plotThresholds(priorThr, norm4, Nds)
                    #print(f"Sum of all possible evidences: {sum(tots)}")
                    #print(f"Nds: {Nds}")
                    #print(f"probas superar Nds: {[1-norm.cdf(0,loc=d.mu, scale=d.sigma2) for d in Nds]}")
            i_umbrales +=1
        if plot:
            Messages.plotThresholds(priorThr, norm4, Nds)
        #evidencias = Messages.evidences(Nds)
        return([math.prod(N11), N12, evidencias])

    @staticmethod
    def plotThresholds(priorThr: list[Gaussian], norm4: Gaussian, Nds: list[Gaussian]) -> None:
        grilla = [x/10.0 for x in range(-100,101,1)]
        plt.plot(grilla,norm4.eval(grilla),label=f"R cont.")
        plt.ylim(0,0.6)
        for i, thr in enumerate(priorThr):
            p = plt.plot(grilla, thr.eval(grilla),label=f"Thr {i}")
            plt.axvline(Nds[i].mu, color=p[0].get_color())
        plt.legend(loc="upper left")
        global PLT_COUNT
        plt.savefig(f"./src/maca/tempfig/thresholds_{PLT_COUNT}.png")
        PLT_COUNT +=1
        plt.clf()
        
    @staticmethod
    def evidencePermutations(ds: list[Gaussian]) -> list[float]:
        over_probs = [norm.cdf(0,loc=d.mu, scale=d.sigma2) for d in ds]
        under_probs = [1-d for d in over_probs]
        maxi = 2**(len(over_probs))
        return [math.prod([under_probs[i] if (len(Messages.mod(n))>i and Messages.mod(n)[i]=="1") else over_probs[i] for i in range(len(over_probs))]) for n in range(maxi)]

    def mod(n):
        if n==0:
            return "0"
        res = ""
        while(n != 0):
            b = n % 2
            res += str(b)
            n = int(n/2)
        return res

    @staticmethod
    def evidences(ds: list[Gaussian]) -> list[float]:
        #sum([math.log(norm.cdf(0,loc=ds[l].mu, scale=ds[l].sigma2)) if l >= rObs else math.log(1-norm.cdf(0,loc=ds[l].mu, scale=ds[l].sigma2)) for l in range(len(priorThr))])
        over_probs = [norm.cdf(0,loc=d.mu, scale=d.sigma2) for d in ds]
        under_probs = [1-d for d in over_probs]
        return [math.prod(under_probs[:i])*math.prod(over_probs[i:]) for i in range(len(ds)+1)]

    @staticmethod
    def six(norm5p: Gaussian, b: Gaussian, normZ: list[Gaussian]) -> list[Gaussian]:
        """Message 6. Update message obtained from noisyRating and bias. Sent to z variable."""
        #zsum = sum(normZ)
        #return [norm5p - b - zsum + nz for nz in normZ]
        res = []
        for k in range(len(normZ)):
            res.append(norm5p - b - sum(normZ[i] for i in range(len(normZ)) if i!=k))
        return res

        res = []
        for k in range(len(normZ)):
            mu6 = norm5p.mu - (b.mu + sum([normZ[i].mu for i in range(len(normZ)) if i != k]))
            sigma26 = norm5p.sigma2 + b.sigma2 + sum([normZ[i].sigma2 for i in range(len(normZ)) if i != k]) #since zk is who the msg is sent to, normZ[k] is not present in the message to sent to it (doesn't participate in estimating itself?)
            res.append(Gaussian(mu6, math.sqrt(sigma26)))
        return res

    @staticmethod
    def seven(norm6: list[Gaussian], margX: list[Gaussian]) -> list[Gaussian]:
        """Proposed approximation of message 7. Outgoing message from multiplication factor to subject variables."""
        assert len(norm6) == len(margX)
        res = []
        for m6, xk in zip(norm6, margX):
            if (isinstance(xk, PointMass)):
                if(isinstance(m6, PointMass)):
                    # Both PointMass
                    if (xk.mu == 0):
                        assert (m6.mu == 0), "The model has zero probability"
                        res.append(Uniform())
                        continue
                    else:
                        res.append(PointMass(m6.mu/xk.mu))
                        continue
                elif(isinstance(m6, Uniform)):
                    # xk is pointMass, m6 is Uniform
                    res.append(m6)
                    continue
                else:
                    # xk is pointMass, m6 is not
                    if xk.mu == 0:
                        res.append(Uniform())
                        continue
                    mu = m6.mu * xk.mu
                    sigma = m6.sigma / xk.mu
                    res.append(PointMass(mu) if sigma==0 else Gaussian(mu, sigma))
                    continue
            elif (isinstance(xk, Uniform) and isinstance(m6, Uniform)):
                res.append(Uniform())
                continue
                # Nota: el caso en que solo una es uniforme parece resolverlo medio raro/mal...
            if(isinstance(m6, PointMass)):
                #m6 is PointMass, xk is not
                assert False, "Variational Message Passing does not support a Product factor with fixed output and two random inputs."
            mu = (m6.mu*xk.mu)/xk.ncsm
            sigma = math.sqrt(m6.sigma2/xk.ncsm)
            if sigma == 0:
                res.append(PointMass(mu))
                continue
            elif sigma == math.inf: 
                res.append(Uniform())
                continue
            else:
                res.append(Gaussian(mu, sigma))          
        return res
    
    @staticmethod
    def predict(contRating, rObs):
        if rObs == 1:
            return(1-norm.cdf(0,loc=contRating.mu, scale=contRating.sigma))
        else:
            return(norm.cdf(0,loc=contRating.mu, scale=contRating.sigma))

    @staticmethod
    def v_w(mu, sigma, margin):
        _alpha = (margin-mu)/sigma
        v = norm.pdf(-_alpha,loc=0,scale=1) / norm.cdf(-_alpha,loc=0,scale=1)
        w = v * (v + (-_alpha))
        return v, w

    @staticmethod
    def trunc(normR: Gaussian, margin: float) -> Gaussian:
        mu = normR.mu
        sigma = normR.sigma
        v, w = Messages.v_w(mu, sigma, margin)
        mu_trunc = mu + sigma * v
        sigma_trunc = sigma * math.sqrt(1-w)
        return Gaussian(mu_trunc, sigma_trunc)
    
    """
    @staticmethod
    def posterior(wprior, likelihoods):
        res = wprior
        for l in likelihoods:
            res = np.multiply(res, l)
        return res
    """
    @staticmethod
    def estimationsConverge(oldEstimation: Gaussian, newEstimation: Gaussian) -> bool:
        muConverge = (oldEstimation.mu - newEstimation.mu) < EPSILON
        sigmaConverge = (oldEstimation.sigma - newEstimation.sigma) < EPSILON
        return muConverge and sigmaConverge

    """
    @staticmethod
    def KLdivergence(x, y, step):
        #assert 0 not in x, "0 will appear in denominator"
        #res = sum([x[i]*np.log(x[i]/y[i])*step for i in range(len(x))])
        #rel_entr(exact, aprox)res = sum([-aprox[i]*np.log(exact[i]/aprox[i])*step for i in range(len(aprox))])
        
        #assert (sum(rel_entr(x, y)*step) == res).all(), "Calculated KL divergence was different from scipy kl div"
        res = sum(rel_entr(np.array(x)*step, np.array(y)*step))
        return res
    
    @staticmethod
    def exactmsg7(norm6, margT, sk, tk):
        exact = []
        for s in sk:
            #probamos agregar logaritmo siguiendo lo que se ve en la documentacion de AAverageLogarithm, que es donde esta implementado
            # el mensaje aproximado 7 de infer.NET. Aca: https://dotnet.github.io/infer/apiguide/api/Microsoft.ML.Probabilistic.Factors.GaussianProductVmpOp.html
            exact.append(sum(np.log(norm6[0].eval([s*t for t in tk]))*margT[0].eval(tk)))
        #exact = [e/(sum(exact)*0.1) for e in exact]
        return exact
    """

    @staticmethod
    def exactmsg2(margS, margT, zk):
        exact = []
        for x in zk:
            exact.append(sum(margS[0].eval(x/zk)*margT[0].eval(zk)))
        exact = exact/(sum(exact)*0.1)
        return exact

    @staticmethod
    def testSeven():
        print(f"N   , N   : {Messages.seven([Gaussian(2,math.sqrt(1.2))], [Gaussian(3,math.sqrt(1.5))])}")
        print(f"N   , P(0): {Messages.seven([Gaussian(2,1.2)], [PointMass(0)])}")
        print(f"P(0), N   : Variational Message Passing does not support a Product factor with fixed output and two random inputs.")
        print(f"P(1), P(0): The model has zero probability")
        print(f"P(0), P(1): {Messages.seven([PointMass(0)], [PointMass(1)])}")
        print(f"P(0), P(0): {Messages.seven([PointMass(0)], [PointMass(0)])}")
        print(f"N   , P(1): {Messages.seven([Gaussian(2,1.2)], [PointMass(1)])}")
        print(f"N   , U   : {Messages.seven([Gaussian(2,1.2)], [Uniform()])}")
        print(f"U   , N   : {Messages.seven([Uniform()], [Gaussian(2,1.2)])}")
        print(f"U   , U   : {Messages.seven([Uniform()], [Uniform()])}")
        exit(0) 
