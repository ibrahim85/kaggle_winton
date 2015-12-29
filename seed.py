def seed(data):
    import numpy as np
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