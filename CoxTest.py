from AgingData import DTtest, get_args
import Splitters as sptr
import numpy as np

def test_PHreg(data):
    from statsmodels.duration.hazard_regression import PHReg
    mod = PHReg(data['ttodeath'], data['sbp'], status=data['death'])
    return mod.fit_regualrized(alpha=np.ones(len(data)), warn_convergence=True)

if __name__ == '__main__':    
    args = get_args()
    test = DTtest(args)
    data = test.data
    if args.sub is not None:
        data = data[:args.sub]

    # mod = sptr.SplitCoef_statsmod(data = data, class_lbl = data['death'], weights = np.ones(len(data)), args = args)
    mod = test_PHreg(data)
    # print(mod.coef, mod.loglikelihood)
    print(mod.summary())