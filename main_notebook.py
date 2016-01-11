from __future__ import division
import numpy as np
import simulateData as sd
import fullyConditionalSpecification as fcs
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

# need this to let Pool work on Windows
if __name__ == '__main__':
      mp.freeze_support()

      # pandas display options
      pd.set_option('precision', 2)
      pd.set_option('display.float_format', lambda x: '%.3f' % x)
      np.set_printoptions(precision=2,suppress=True)

      # simulate data
      nObs = 1000
      nFeatures = 5
      data_complete = sd.simulateData(nObs, nFeatures)
      (n, f) = data_complete.shape
      print 'The dataset has %.0f observations and %.0f features' % (n, f)

      # NaN out a subset
      data = sd.randomlyNanOutSubsetOfData(data_complete, 0.15)
      print 'The fraction of the data that is missing is %.2f' % \
            (sum(np.ravel(np.isnan(data))) / data.size)
      nidx = np.isnan(data)

      # test fullyConditionalSpecification module
      # seed the data from each feature's univariate distribution
      data_seeded = fcs.seed(data)
      # predict each feature with the others, using the method specified
      data_predicted, scores = fcs.predict(data_seeded, 'ols')
      # replace original NaNs with forecasts, leave non-NaN data untouched
      data_updated = np.copy(data)
      data_updated[nidx] = data_predicted[nidx]
      print np.corrcoef(data_updated,data_complete,rowvar=False)

      # again for extra trees
      data_predicted_extra, scores_extra = fcs.predict(data_seeded, 'extra')
      data_updated_extra = np.copy(data)
      data_updated_extra[nidx] = data_predicted_extra[nidx]
      print np.corrcoef(data_updated_extra,data_complete,rowvar=False)


      rmse_seed    = np.sqrt(np.sum((data_seeded-data_complete)**2))
      rmse_fcs_lin = np.sqrt(np.sum((data_updated-data_complete)**2))
      rmse_fcs_extra = np.sqrt(np.sum((data_updated_extra-data_complete)**2))
      # rmse_knn     = np.sqrt(np.sum((datahat_knn-data)**2))
      # rmse_fcs_sgd = np.sqrt(np.sum((datahat_fcs_sgd-data)**2))
      fig, ax = plt.subplots()
      ax.bar([1,2,3],[rmse_seed,rmse_fcs_lin,rmse_fcs_extra],width=0.75,color='b',align='center')
      # ax.bar([1,2,3],[rmse_knn,rmse_fcs_lin,rmse_fcs_sgd],width=0.75,color='b',align='center')
      ax.set_title('RMSE by method')
      ax.set_ylabel('RMSE')
      ax.set_xticks([1,2,3])
      ax.set_xticklabels(('seed','FCS (ols)','FCS (extra)'))
      # ax.set_xticklabels(('KNN', 'FCS (ols)','FCS (sgd)'))
      plt.show()




