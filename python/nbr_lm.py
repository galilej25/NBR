import csv
import time
import signal
import numpy as np
import pandas as pd
import scipy.stats
import multiprocessing as mp
import statsmodels.formula.api as smf


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def fit_linear_model(data, dependent, formula):
    """

    :param data:
    :param dependent:
    :param formula:
    :return:
    """
    # create a linear model
    # FIXME: we need to figure out a better way to parse formula
    lm = smf.ols(
        formula='%s ~ %s' % (dependent, formula),
        data=data
    )
    res = lm.fit()

    # return t-values and p-values for each independent variable
    res_out = pd.concat([res.tvalues, res.pvalues], axis=1).iloc[1:]
    res_out.columns = ['tvals_%s' % dependent, 'pvals_%s' % dependent]
    return res_out


def get_observations(data, mod, thr_t, thr_p, alternative, ci):
    """

    :return:
    """
    st = time.time()
    lms = []
    for column in data.columns[3:]:
        lms.append(
            fit_linear_model(
                data,
                column,
                mod
            )
        )
    lms = pd.concat(lms, axis=1)
    et = time.time()
    print('Linear Models Estimation finished in %.5f [s]' % (et-st))

    st = time.time()
    df_tvals = lms[[col for col in lms.columns if 'tvals_' in col]]
    df_pvals = lms[[col for col in lms.columns if 'pvals_' in col]]
    qt_2 = scipy.stats.t.ppf(1 - thr_p / 2, data.shape[0] - lms.shape[0] - 1)
    qt = scipy.stats.t.ppf(1 - thr_p, data.shape[0] - lms.shape[0] - 1)

    if thr_t is not None:
        # calculate strength
        strength = (df_tvals.abs() - abs(thr_t)) * (1 - 2 * (df_tvals < 0))

        if alternative == "two.sided":
            edges = df_tvals.abs() > thr_t
        elif alternative == "less":
            edges = df_tvals < thr_t
        else:
            edges = df_tvals > thr_t
    else:
        if alternative == "two.sided":
            edges = df_pvals < thr_p
            strength = (df_tvals.abs() - qt_2) * (1 - 2 * (df_tvals < 0))
        elif alternative == "less":
            edges = pd.DataFrame((df_pvals < 2 * thr_p).values & (df_tvals < 0).values, columns=df_pvals.columns,
                                 index=df_pvals.index)
            strength = (df_tvals.abs() - qt) * (1 - 2 * (df_tvals < 0))
        else:
            edges = pd.DataFrame((df_pvals < 2 * thr_p).values & (df_tvals > 0).values, columns=df_pvals.columns,
                                 index=df_pvals.index)
            strength = (df_tvals.abs() - qt) * (1 - 2 * (df_tvals < 0))
    et = time.time()
    print('Edges and strength calculation finished in %.5f [s]' % (et-st))

    st = time.time()
    observations = {}
    for var in edges.index:
        e = np.where(edges.loc[var] == 1)
        ci_row = ci[e, 0]
        ci_col = ci[e, 1]
        s = strength.loc[var].values[e]
        observations[var] = pd.DataFrame(np.vstack((e, ci_row, ci_col, s)).T,
                                         columns=["2Dcol", "3Drow", "3Dcol", "strn"])
    et = time.time()
    print('Saving clusters for each variable finished in %.5f [s]' % (et-st))

    return observations


class NBR(object):
    """
    NBR Linear Model (2D input)

    Parameters
    ----------

    """

    def __init__(
            self,
            fname,
            n_nodes,
            mod,
            alternative,
            n_perm,
            diag=False,
            thr_p=0.05,
            thr_t=None,
            cores=None,
            nudist=False,
            exp_list=None,
            verbose=True,
        ):

        # FIXME: What about missing values, etc...
        try:
            # load data
            self.data = pd.read_csv(
                fname,
                sep=",",
                header=0
            ).apply(pd.to_numeric, errors='ignore')
            # convert dots from header into slash
            self.data.columns = [c.replace('.', '_') for c in self.data.columns]
        except FileNotFoundError:
            print('[error] File Not Found!')

        self.n_nodes = n_nodes
        self.mod = mod
        self.alternative = alternative
        self.n_perm = n_perm
        self.diag = diag
        self.thr_p = thr_p
        self.thr_t = thr_t
        self.nudist = nudist
        self.expList = exp_list
        self.verbose = verbose

        if cores is None:
            self.cores = 1
        elif cores == -1:
            self.cores = mp.cpu_count()
        else:
            self.cores = cores

        self._args_check()

    def _args_check(self):
        """
        Check correct type of arguments
        :return:
        """

        # check if diag is boolean
        if not isinstance(self.diag, bool):
            raise ValueError("[error] diag parameter must be boolean; got (diag=%r)" % self.diag)

        if not isinstance(self.n_nodes, int) and self.n_nodes > 0:
            raise ValueError("[error] n_nodes parameter must be positive non-zero integer; got (n_nodes=%r)" % self.n_nodes)

        if not isinstance(self.n_perm, int) and self.n_perm > 0:
            raise ValueError("[error] n_perm parameter must be positive non-zero integer; got (n_perm=%r)" % self.n_perm)

        if self.alternative not in ["two.sided", "less", "greater"]:
            raise ValueError("[error] alternative parameter is not correctly specified (options: two.sided, less, greater); got (alternative=%r) " % self.alternative)

        # FIXME: no need to check this if we are already processing it through reading file
        # if len(self.net) != len(self.idata):
        #     raise ValueError("Input data dimensions missmatch;")

        if (self.thr_p is None and self.thr_t is None) or (self.thr_t is not None and self.thr_p is not None):
            raise ValueError('Thresholds for P and T are incorrectly set; got(pval=%r, tval=%r)' % (self.thr_p, self.thr_t))

    def _create_connection_indices(self):
        """

        :return:
        """
        cn = np.vstack((np.where(np.triu(np.ones(self.n_nodes), k=0) == 0))).T
        cn[:, [0, 1]] = cn[:, [1, 0]]
        self.connection_indices = cn

    def _fit_linear_model(self):

        # 1. create connection indicies
        self._create_connection_indices()

        get_observations(self.data, self.mod, self.thr_t, self.thr_p, self.alternative, self.connection_indices)

    def _permutations(self, n_core, n_perm):
        st = time.time()
        self._create_connection_indices()
        pool = mp.Pool(n_core, init_worker)
        results_p = []
        try:
            for _ in range(n_perm):
                df_ = pd.concat([self.data[['Group', 'Sex', 'Age']].sample(self.data.shape[0]).reset_index(drop=True),
                                 self.data.drop(columns=['Group', 'Sex', 'Age'])], axis=1)
                results_p.append(pool.apply_async(get_observations, args=(self.data, self.mod, None, 0.05, "less", self.connection_indices)))
        except KeyboardInterrupt:
            print('[Exception: Keyboard] Terminate MultiProcessing')
            pool.terminate()
            pool.join()
        except Exception as e:
            print('[Exception: Unknown] Terminate MultiProcessing')
            pool.terminate()
            pool.join()
        finally:
            # Close and wait for the pool to finish
            pool.close()
            pool.join()
            results_p = [r.get() for r in results_p]

        et = time.time()
        print('[perfr-measurment] %d permutations with ncore=%d finished in %.2f [sec]' % (n_perm, n_core, et - st))