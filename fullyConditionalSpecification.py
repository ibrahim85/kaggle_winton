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
    pooledResults = pool.map(partial(predictOneColumn, data, fitmethod), range(nFeatures))
    pool.close()
    pool.join()

    # extract and format the pooled results
    forecastedDataList = [pooledResult[0] for pooledResult in pooledResults]
    data_forecasted = np.column_stack(forecastedDataList)
    scores = [pooledResult[1] for pooledResult in pooledResults]

    # return forecasts for the entire dataset
    # the calling program will use this to update data that were originally NaN
    # (which this function can't see)
    return data_forecasted, scores


def predictOneColumn(data,fitmethod,yindex):
    """forecast one variable using the others"""
    # yindex is the column to forecast using the other columns

    from sklearn import linear_model, cross_validation, ensemble

    # we can implement as many fit methods as we like
    # next up: ExtraTreesRegressor
    fitmethods = dict(ols=linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True),
                      extra=ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None,
                                                         min_samples_split=2, min_samples_leaf=1,
                                                         min_weight_fraction_leaf=0.0, max_features='auto',
                                                         max_leaf_nodes=None, bootstrap=False, oob_score=False,
                                                         n_jobs=1))

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