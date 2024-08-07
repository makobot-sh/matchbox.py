import math
from Gaussian import Gaussian
from Gaussian import PointMass
import sys
from scipy.stats import norm
import random
import pandas as pd
from tqdm import tqdm
from Matchbox import Matchbox
import numpy as np

assert sys.version_info >= (3, 9), "Use Python 3.9 or newer"

def formatNumToString(var):
    if math.isinf(var):
        var = "np.inf" if var > 0 else "-np.inf"
    else:
        var = f"{var:.5f}"
    return var

def GaussianToString(g):
    if g.IsPointMass:
        return f"Gaussian.PointMass({formatNumToString(g.Point)})"
    else:
        return f"Gaussian({formatNumToString(g.GetMean())}, {formatNumToString(g.GetVariance())})"  

def addEstimatedValues(df, user, thr=None, tra=None, bias=None):
    if thr is not None:
        for i in range(len(thr)):
            df.loc[user, f'Threshold_{i}'] = thr[i]
    if tra is not None:
        for i in range(len(tra)):
            df.loc[user, f'Trait_{i}'] = tra[i]
    if bias is not None:
        df.loc[user, f'Bias'] = bias
    return df

def generateThresholds(numThresholds):
    ##if numThresholds == 1:
    ##    res.append([PointMass(0)])
    ##    continue
    thresholds = [Gaussian(l - numThresholds / 2.0 + 0.5, 1.0) for l in range(numThresholds)]
    if numThresholds%2 == 1:
        thresholds[int((numThresholds+1)/2)-1] = PointMass(0) #Matchbox seems to fix side and middle thresholds in -inf, 0, +inf
    return [t.sample() for t in thresholds]

def generateItems(numItems, numTraits, a=0.5, b=0.5):
    res = []
    for i in range(numItems):
        """
        ls = beta.rvs(a, b, size=numTraits)
        total_percentages = ((ls/np.sum(ls)))
        traits = total_percentages * numTraits #Why do this?
        traits *= np.array([-1 if x==1 else 1 for x in binom.rvs(n=1, p=0.5, size=numTraits)])
        """
        traits = norm.rvs(loc=0, scale=1, size=numTraits)
        res.append(traits)
    return res

def GenerateData(path, numUsers, numItems, numTraits, numObs, numThresholds):
    affinityNoiseVariance = 1
    thresholdNoiseVariance = 0

    itemTraits = generateItems(numItems, numTraits)
    for i in range(numTraits): # Break symmetry
        itemTraits[i][0:i] = 0
        itemTraits[i][i] = 1
        itemTraits[i][i+1:] = 0
    userTraits = generateItems(numUsers, numTraits)
    itemBias = norm.rvs(0,1,size=numItems)
    userBias = norm.rvs(0,1,size=numUsers)
    UserThresholds = [generateThresholds(numThresholds) for _ in range(numUsers)]

    generated_users = pd.DataFrame.from_dict({"User":list(range(numUsers))})
    for user in range(numUsers):
        generated_users = addEstimatedValues(generated_users, user, thr=UserThresholds[user], tra=userTraits[user])
    generated_users["Bias"] = userBias

    generated_items = pd.DataFrame.from_dict({"Item":list(range(numItems))})
    for item in range(numItems):
        generated_items = addEstimatedValues(generated_items, item, thr=None, tra=itemTraits[item])
    generated_items["Bias"] = itemBias

    generatedUserData = []
    generatedItemData = []
    generatedRatingData = []
    
    visited = set()
    iObs = 0

    with tqdm(total=numObs) as pbar:
        while iObs < numObs:
            user = random.randrange(numUsers)
            item = random.randrange(numItems)
            if iObs < max(numTraits,numItems):
                item = iObs
            userItemPairID = user * numItems + item #pair encoding  

            if userItemPairID in visited: #duplicate generated
                continue #redo this iteration with different user-item pair

            visited.add(userItemPairID);
            
            products = np.array(userTraits[user]) * np.array(itemTraits[item])
            bias = userBias[user] + itemBias[item]
            affinity = bias + np.sum(products)
            noisyAffinity = norm.rvs(affinity, affinityNoiseVariance)
            noisyThresholds = [norm.rvs(ut, thresholdNoiseVariance) for ut in UserThresholds[user]]

            generatedUserData.append(user);  
            generatedItemData.append(item)
            generatedRatingData.append([1 if noisyAffinity > noisyThresholds[l] else 0 for l in range(len(noisyThresholds))])
            iObs += 1
            pbar.update(1)

    df = pd.DataFrame.from_dict({"user":generatedUserData, "item":generatedItemData, "ratingList":generatedRatingData})
    df["rating"] = [np.sum(r) for r in generatedRatingData]
    #df["timestamps"] = [999 for _ in range(len(generatedRatingData))]

    generated_users.to_csv(f"{path}/user_truth.csv", header=True, index=False)
    generated_items.to_csv(f"{path}/item_truth.csv", header=True, index=False)
    df[["user","item","rating"]].to_csv(f"{path}/ratings_train.csv", header=False, index=False)

    return df[["user","item","rating"]], generated_users, generated_items

def GeneratedDataTest():
    path = "./src/maca/temp"
    numTraits = 2
    numThresholds = 3
    largeData = False
    numUsers = 400 if largeData else 10#50
    numItems = 400 if largeData else 10#10
    numObs = int(numUsers*numItems/2)
    ratings, users_t, items_t = GenerateData(path, numUsers, numItems, numTraits, numObs, numThresholds)

    recommender = Matchbox()
    recommender.numTraits = numTraits
    recommender.numThresholds = numThresholds
    for idx, row in tqdm(ratings.iterrows(), total=ratings.shape[0]):
        recommender.addRating(row["user"], row["item"], row["rating"])
    
    recommender.convergeModel()
    recommender.printEvidence()

    items = pd.DataFrame.from_dict(recommender.V, orient="index", columns=[f"Trait_{i}_est" for i in range(numTraits)])
    itemBias = pd.DataFrame.from_dict(recommender.vbias, orient="index", columns=["Bias_est"])
    users = pd.DataFrame.from_dict(recommender.U, orient="index", columns=[f"Trait_{i}_est" for i in range(numTraits)])
    userBias = pd.DataFrame.from_dict(recommender.ubias, orient="index", columns=["Bias_est"])
    user_thr = pd.DataFrame.from_dict(recommender.U_thr, orient="index", columns=[f"Threshold_{i}_est" for i in range(numThresholds)])
    
    col_order = [format_string.format(i) for i in range(numTraits) for format_string in ["Trait_{}","Trait_{}_est"]] + ["Bias","Bias_est"]
    col_order_user = ["User"] + col_order + [format_string.format(i) for i in range(numThresholds) for format_string in ["Threshold_{}","Threshold_{}_est"]]
    col_order_item = ["Item"] + col_order
    items_est = items.join(itemBias)
    users_est = users.join([userBias, user_thr])

    items_est.to_csv(f"{path}/item_estimated.csv", header=True, index=False)
    users_est.to_csv(f"{path}/user_estimated.csv", header=True, index=False)

    print("Item results:")
    item_res = items_t.join([items,itemBias])[col_order_item]
    print(item_res)
    item_res.to_csv(f"{path}/item_results.csv", header=True, index=False)

    print("User results:")
    user_res = users_t.join([users,userBias,user_thr])[col_order_user]
    print(user_res)
    user_res.to_csv(f"{path}/user_results.csv", header=True, index=False)

if __name__ == "__main__":
    with open("./src/maca/ratings_big.txt") as ratings:
        path = "./src/maca/temp"
        numTraits = 2
        numThresholds = 3
        largeData = False
        numUsers = 400 if largeData else 50
        numItems = 400 if largeData else 10
        numObs = int(numUsers*numItems/2)

        recommender = Matchbox()
        recommender.numTraits = numTraits
        recommender.numThresholds = numThresholds
        for idx, row in tqdm(enumerate(ratings), total=numObs):
            recommender.addRating(*[int(rat) for rat in row.split(",")])
        
        recommender.convergeModel()
        recommender.printEvidence()

        items = pd.DataFrame.from_dict(recommender.V, orient="index", columns=[f"Trait_{i}_est" for i in range(numTraits)])
        itemBias = pd.DataFrame.from_dict(recommender.vbias, orient="index", columns=["Bias_est"])
        users = pd.DataFrame.from_dict(recommender.U, orient="index", columns=[f"Trait_{i}_est" for i in range(numTraits)])
        userBias = pd.DataFrame.from_dict(recommender.ubias, orient="index", columns=["Bias_est"])
        user_thr = pd.DataFrame.from_dict(recommender.U_thr, orient="index", columns=[f"Threshold_{i}_est" for i in range(numThresholds)])
        
        col_order = [format_string.format(i) for i in range(numTraits) for format_string in ["Trait_{}","Trait_{}_est"]] + ["Bias","Bias_est"]
        col_order_user = ["User"] + col_order + [format_string.format(i) for i in range(numThresholds) for format_string in ["Threshold_{}","Threshold_{}_est"]]
        col_order_item = ["Item"] + col_order
        items_est = items.join(itemBias)
        users_est = users.join([userBias, user_thr])

        items_est.to_csv(f"{path}/item_estimated.csv", header=True, index=False)
        users_est.to_csv(f"{path}/user_estimated.csv", header=True, index=False)
        print(items_est)
        print(users_est)
