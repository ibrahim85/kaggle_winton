def fcs(data):
    import multiprocessing as mp
    import numpy as np
    from functools import partial

    # data has already been seeded; it contains no NaNs
    nFeatures = data.shape[1]

    # start a pool of multiprocessors
    # number of processes defaults to number of cores on your machine
    pool = mp.Pool(processes=min(nFeatures, mp.cpu_count()))

    # give the same dataset to each processor, but indexing a different column to forecast
    forecastedDataList = pool.map(partial(fcsOneColumn, data), range(nFeatures))

    # do some cleanup (what does join do?)
    pool.close()
    pool.join()

    # update the dataset's missing values with the forecasts
    data_updated = np.copy(data)
    for f in range(nFeatures):
        nidx_f = np.isnan(data[:, f])
        data_updated[nidx_f, f] = forecastedDataList[f][nidx_f]

    return data_updated
