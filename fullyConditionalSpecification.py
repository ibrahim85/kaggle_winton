import numpy as np


def predict(data,fitmethod):
    """ fitmethod can be 'ols' or 'extra' (for extremely randomized forest) """

    import multiprocessing as mp
    from functools import partial

    # data has already been seeded; it contains no NaNs
    nFeatures = data.shape[1]

    # start a pool of multiprocessors
    # number of processes defaults to number of cores on your machine
    pool = mp.Pool(processes=min(nFeatures, mp.cpu_count()))

    # give the same dataset to each processor, but indexing a different column to forecast
    pooledResults = pool.map(partial(fcsOneColumn, data, fitmethod), range(nFeatures))
    forecastedDataList = [pooledResult[0] for pooledResult in pooledResults]
    scores = [pooledResult[1] for pooledResult in pooledResults]

    # do some cleanup (what does join do?)
    pool.close()
    pool.join()

    # update the dataset's missing values with the forecasts
    data_updated = np.copy(data)
    for f in range(nFeatures):
        nidx_f = np.isnan(data[:, f])
        data_updated[nidx_f, f] = forecastedDataList[f][nidx_f]

    return data_updated, scores


def predictOneColumn(data,fitmethod,yindex):
    """forecast one variable using the others"""
    # yindex is the column to forecast using the other columns

    from sklearn import linear_model, cross_validation

    # we can implement as many fit methods as we like
    # next up: ExtraTreesRegressor
    fitmethods = {'ols':linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True),
                  'sgd':linear_model.SGDRegressor(fit_intercept=True)}

    # X is every column except the column specified by yindex
    ncols = data.shape[1]
    X = data[:,[c for c in range(ncols) if c != yindex]]
    y = data[:,yindex]

    # Cross-Validation Score
    scores = cross_validation.cross_val_score(fitmethods[fitmethod], X, y, cv=5)

    # Prediction
    fitmethods[fitmethod].fit(X,y)
    yhat = fitmethods[fitmethod].predict(X)

    # return the forecasts and the mean of the cross-validation scores
    return yhat, scores.mean()


def seed(data):
    from scipy import stats

    nFeatures = data.shape[1]
    kdes = []
    data_seeded = np.copy(data)
    for i in range(nFeatures):
        nidx_i = np.isnan(data[:,i])

        # we need to get the sample distribution of each variable
        kde_i = stats.gaussian_kde(data[~nidx_i,i])
        kdes.append(kde_i)

        # draw randomly from the sample distribution to fill in the missing values
        sample_i = kdes[i].resample(sum(nidx_i)).T

        # stupid reshaping due to broadcasting error
        data_seeded[nidx_i,i] = sample_i.reshape(len(sample_i),)

    return data_seeded