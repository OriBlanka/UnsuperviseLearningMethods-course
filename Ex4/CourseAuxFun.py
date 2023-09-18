import numpy as np
from scipy.stats import multivariate_normal as MVN


def InitKMeans(mX: np.ndarray, K: int, initMethod: int = 0, seedNum: int = 123) -> np.ndarray:
    '''
    K-Means algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        initMethod  - Initialization method: 0 - Random, 1 - K-Means++.
        seedNum     - Seed number used.
    Output:
        mC          - The initial centroids with shape K x d.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
    '''

    np.random.seed(seedNum)
    if not initMethod:  # random

        return mX[np.random.choice(mX.shape[0], K, replace=False), :]
    # choose by K-Means++
    mC = np.zeros((K, mX.shape[1]))
    samples_chosen_idxs = []
    samples_chosen_idx = np.random.choice(mX.shape[0], 1, replace=False)
    mC[0] = mX[samples_chosen_idx, :]
    samples_chosen_idxs.append(samples_chosen_idx)
    for i in range(1, K):
        mD = np.zeros((mX.shape[0], i))
        for j in range(i):
            mD[:, j] = np.linalg.norm(mX - mC[j], axis=1)

        # Validate that we are not choosing the same sample twice
        mD[samples_chosen_idxs, :] = 0
        # find the maximum value in each row
        mDmax = np.max(mD, axis=1)
        mDmax /= np.sum(mDmax)
        samples_chosen_idx = np.random.choice(
            mX.shape[0], 1, replace=False, p=mDmax)
        mC[i] = mX[samples_chosen_idx, :]
        samples_chosen_idxs.append(samples_chosen_idx)
    return mC


def CalcKMeansObj(mX: np.ndarray, mC: np.ndarray) -> float:
    '''
    K-Means algorithm.
    Args:
        mX          - The data with shape N x d.
        mC          - The centroids with shape K x d.
    Output:
        objVal      - The value of the objective function of the KMeans.
    Remarks:
        - The objective function uses the squared euclidean distance.
    '''

    mD = np.zeros((mX.shape[0], mC.shape[0]))
    for i in range(mC.shape[0]):
        mD[:, i] = np.linalg.norm(mX - mC[i], axis=1)
    return np.sum(np.min(mD, axis=1) ** 2)


def KMeans(mX: np.ndarray, mC: np.ndarray, numIter: int = 1000, stopThr: float = 0) -> np.ndarray:
    '''
    K-Means algorithm.
    Args:
        mX          - Input data with shape N x d.
        mC          - The initial centroids with shape K x d.
        numIter     - Number of iterations.
        stopThr     - Stopping threshold.
    Output:
        mC          - The final centroids with shape K x d.
        vL          - The labels (0, 1, .., K - 1) per sample with shape (N, )
        lO          - The objective value function per iterations (List).
    Remarks:
        - The maximum number of iterations must be `numIter`.
        - If the objective value of the algorithm doesn't improve by at least `stopThr` the iterations should stop.
    '''

    lO = []
    for i in range(numIter):
        mD = np.zeros((mX.shape[0], mC.shape[0]))
        for j in range(mC.shape[0]):
            mD[:, j] = np.linalg.norm(mX - mC[j], axis=1)
        vL = np.argmin(mD, axis=1)
        lO.append(CalcKMeansObj(mX, mC))
        if i > 0 and abs(lO[-1] - lO[-2]) < stopThr:
            return mC, vL, lO
        for j in range(mC.shape[0]):
            mC[j] = np.mean(mX[vL == j], axis=0)
    return mC, vL, lO


def InitGmm(mX: np.ndarray, K: int, seedNum: int = 123) -> np.ndarray:
    '''
    GMM algorithm initialization.
    Args:
        mX          - Input data with shape N x d.
        K           - Number of clusters.
        seedNum     - Seed number used.
    Output:
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Remarks:
        - Given the same parameters, including the `seedNum` the algorithm must be reproducible.
        - mμ Should be initialized by the K-Means++ algorithm.
    '''

    mμ = InitKMeans(mX, K, initMethod= 1, seedNum= seedNum)
    vW = np.ones(K) / K
    tΣ = np.zeros((mX.shape[1], mX.shape[1], K))
    for i in range(K):
        tΣ[:, :, i] = np.eye(mX.shape[1])
    return mμ, tΣ, vW

def CalcGmmObj(mX: np.ndarray, mμ: np.ndarray, tΣ: np.ndarray, vW: np.ndarray) -> float:
    '''
    GMM algorithm objective function.
    Args:
        mX          - The data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
    Output:
        objVal      - The value of the objective function of the GMM.
    Remarks:
        - A
    '''

    objVal = 0
    for i in range(mX.shape[0]):
        for j in range(mμ.shape[0]):
            objVal += -np.log(vW[j] * MVN.pdf(mX[i], mμ[j], tΣ[:, :, j]))
    return objVal


def GMM(mX: np.ndarray, mμ: np.ndarray, tΣ: np.ndarray, vW: np.ndarray, numIter: int = 1000, stopThr: float = 1e-5) -> np.ndarray:
    '''
    GMM algorithm.
    Args:
        mX          - Input data with shape N x d.
        mμ          - The initial mean vectors with shape K x d.
        tΣ          - The initial covariance matrices with shape (d x d x K).
        vW          - The initial weights of the GMM with shape K.
        numIter     - Number of iterations.
        stopThr     - Stopping threshold.
    Output:
        mμ          - The final mean vectors with shape K x d.
        tΣ          - The final covariance matrices with shape (d x d x K).
        vW          - The final weights of the GMM with shape K.
        vL          - The labels (0, 1, .., K - 1) per sample with shape (N, )
        lO          - The objective function value per iterations (List).
    Remarks:
        - The maximum number of iterations must be `numIter`.
        - If the objective value of the algorithm doesn't improve by at least `stopThr` the iterations should stop.
    '''

    lO = []
    vL = np.zeros(mX.shape[0])
    for i in range(numIter):
        # Step 1
        pXI_K = np.zeros((mX.shape[0], mμ.shape[0]))
        for j in range(mX.shape[0]):
            pXI_K[j, :] = [vW[k] * MVN.pdf(mX[j], mμ[k], tΣ[:, :, k] ) for k in range(mμ.shape[0])]
            pXI_K[j, :] /= pXI_K[j, :].sum()

        vL = pXI_K.argmax(axis=1)
        # Step 2
        for k in range(mμ.shape[0]):
            nK = pXI_K[:, k].sum()
            vW[k] = nK / mX.shape[0]
            mμ[k] = 1/nK * (pXI_K[:, k].reshape(-1, 1) * mX).sum(axis=0)
            tΣ[:, :, k] = 1/nK * (pXI_K[:, k].reshape(-1, 1, 1) * (mX - mμ[k]).reshape(-1, 2, 1) * (mX - mμ[k]).reshape(-1, 1, 2)).sum(axis=0)
            
        lO.append(CalcGmmObj(mX, mμ, tΣ, vW))
        if i > 0 and abs(lO[-1] - lO[-2]) < stopThr:
            break
    return mμ, tΣ, vW, vL, lO