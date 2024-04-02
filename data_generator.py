#!/usr/bin/env python
# coding: utf-8

# In[2]:


from learn import em_learn, svd_learn_new
from data import *
import numpy as np



import numpy as np
from hmmlearn import hmm



# In[3]:


def svd_learn(sample, n, L=None, verbose=None, stats={}):
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)

    svds = [np.linalg.svd(Os[j], full_matrices=True) for j in range(n)]

    if verbose:
        for i, (_, s, _) in enumerate(svds):
            print(f"{i}: {s[:L+1]} ...")

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0 : min(n, L), :] = u[:, 0:L].T
        Qs_[j, 0 : min(n, L), :] = (np.diag(s) @ (vh))[0:L, :]

    A = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L * j : L * (j + 1), n * j : n * (j + 1)] = Ps_[j]
        A[L * (n + j) : L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    small = list(s < 1e-5)
    if True in small:
        fst = small.index(True)
        if verbose:
            print(2 * L * n - fst, L, s[[fst - 1, fst]])
    B = vh[-L:]
    Bre = np.moveaxis(B.reshape((L, L, 2 * n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n : 2 * n]

    Xs = [
        np.linalg.pinv(Zs_[j] @ Ys_[j].T) @ (Zs_[j + 1] @ Ys_[j + 1].T)
        for j in range(n - 1)
    ]
    X = np.sum(Xs, axis=0)
    _, R_ = np.linalg.eig(X)
    d, _, _, _ = np.linalg.lstsq(
        (R_.T @ Ys_[0] @ Ps_[0]).T, Os[0] @ np.ones(n), rcond=None
    )

    R = np.diag(d) @ R_.T
    Ys = R @ Ys_
    Ps = np.array([Y @ P_ for Y, P_ in zip(Ys, Ps_)])
    Ss = np.array([R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_)])

    S_ = np.zeros((L, n))
    Ms_ = np.zeros((L, n, n))
    for l in range(L):
        for i in range(n):
            S_[l, i] = Ss[i, l, l]
            for j in range(n):
                Ms_[l, i, j] = Ps[j, l, i] / S_[l, i]

    S_ = np.abs(S_)
    Ms_ = np.abs(Ms_)
    learned_mixture = Mixture(S_, Ms_)
    learned_mixture.normalize()
    return learned_mixture

learners = {
    "CA-SVD": svd_learn_new,
    "CA-SVD'": lambda d, n, L: svd_learn_new(d, n, L, sample_dist=0.01),
    "GKV-SVD": svd_learn,
    "EM2": lambda d, n, L: em_learn(d, n, L, max_iter=2),
    "EM5": lambda d, n, L: em_learn(d, n, L, max_iter=5),
    "EM20": lambda d, n, L: em_learn(d, n, L, max_iter=20),
    "EM50": lambda d, n, L: em_learn(d, n, L, max_iter=50),
    "EM100": lambda d, n, L: em_learn(d, n, L, max_iter=100),
    "EM-converge": em_learn,
    "CA-SVD-EM2": lambda d, n, L: svd_learn_new(d, n, L, em_refine_max_iter=2),
    "CA-SVD-EM5": lambda d, n, L: svd_learn_new(d, n, L, em_refine_max_iter=5),
    "CA-SVD-EM20": lambda d, n, L: svd_learn_new(d, n, L, em_refine_max_iter=20),
    "CA-SVD-EM100": lambda d, n, L: svd_learn_new(d, n, L, em_refine_max_iter=100),
}

def count_3_from_seq(seq, n):
    """
    seq: discretized sequence
    n: number of categories
    """
    all_trail_probs = np.zeros((n, n, n)) 
    for i in range(len(seq) // 3):
        x = seq[3*i:3*(i+1)]
        all_trail_probs[tuple(x)] += 1
       #num_visited[x] += 1
    return Distribution.from_all_trail_probs(all_trail_probs / np.sum(all_trail_probs))
    
def learn_mix_from_seq(seq,learner, n, L):
    """
    seq: discretized time series: an 1-d array
    learner: 
    """
    trail_empirical_distribution = count_3_from_seq(seq, n)
    if np.isnan(trail_empirical_distribution.all_trail_probs()).any() or np.isinf(trail_empirical_distribution.all_trail_probs()).any():
        print("Inf or NAN values")
        print(trail_empirical_distribution.all_trail_probs())
        
    return  learners[learner](trail_empirical_distribution, n, L)

def likelihood(mixture, trails, counts=None, log=False):
    if counts is None: counts = transitions(mixture.n, trails)
    logS = np.log(mixture.S + 1e-10)
    logTs = np.log(mixture.Ms + 1e-10)

    logl = logS[:, trails[:,0]]
    logl += np.sum(logTs[:, :, :, None] * np.moveaxis(counts, 0, 2)[None, :, :, :], axis=(1,2))
    if log: return logl
    probs = np.exp(logl - np.max(logl, axis=0))
    probs /= np.sum(probs, axis=0)[None, :]
    return probs

def transitions(n, trails):
    n_samples = trails.shape[0]
    c = np.zeros([n_samples, n, n], dtype=int)
    for t, trail in enumerate(trails):
        i = trail[0]
        for j in trail[1:]:
            c[t, i, j] += 1
            i = j
    return c


# In[4]:


n_states = 10
L_chains = 5
current_state = 3
mix = Mixture.random(n_states, L_chains)
xs = mix.sample(1,int(1e5), length = int(1e6)).astype(int)


# In[5]:


num_categories = n_states
window = len(xs)//100
L = 5
correct_count = 0
error = []
neg_ll = []
predict = []
for i in range(1000):
    subseq = xs[i:i+window]
    learned_mix = learn_mix_from_seq(subseq,'GKV-SVD', num_categories, L)
    chain_prob = likelihood(learned_mix, np.atleast_2d(subseq[-2:]))
    #most_likely_index = np.argmax(chain_prob)
    #multi_dim_index = np.unravel_index(most_likely_index, learned_mix.S.shape)
    # Based on likelihood probability to find the most likely chain.
    most_likely_chain = np.argmax(chain_prob)
    prob_next_step = learned_mix.Ms[most_likely_chain, subseq[window - 1], :]
    neg_log_likelihood = -np.log(prob_next_step[xs[i + window]]) + np.log(np.max(prob_next_step))
    sorted_indices = np.argsort(prob_next_step)
    predict.append(np.argmax(prob_next_step))
    rank = np.where(sorted_indices == xs[i + window])[0][0]

    neg_ll.append(neg_log_likelihood)
    error.append(59 - rank)


# In[111]:


predict = np.array(predict)
print(abs(predict - xs[window :window + 1000]).mean())
print((predict == xs[window:window + 1000]).astype(int).mean())

# The more chains, the less accurate


# In[14]:


error = np.array(error)
error.mean()


# In[ ]:





# In[ ]:


print(predict == xs)


# In[47]:


import statsmodels.api as sm
import pandas as pd


# In[48]:


#endog = pd.read_csv('energydata_complete.csv')['RH_5']
endog = pd.DataFrame(xs, columns=['self_generate'])
# We could also fit a more complicated model with seasonal components.
# As an example, here is an SARIMA(1,1,1) x (0,1,1,4):


# In[112]:


# num_categories = 60
# n = num_categories

# all_trail_probs = np.zeros((n, n, n))
# num_visited = np.zeros(num_categories)

# df = pd.read_csv('energydata_complete.csv')
# # consider one column first
# #df = df['RH_5']
# #xs = pd.cut(df, bins=num_categories, labels=False)
# res = pd.qcut(df['RH_5'],n, labels=False, retbins=True, precision=3, duplicates='raise')
# # do equal-depth p
# xs = np.array(list(res[0]))
# bins = res[1]


predict_sarima = []
for i in range(100):
    subseq = xs[i:i+window]
    df = pd.DataFrame(subseq)

    mod_sarimax = sm.tsa.SARIMAX(subseq, order=(1,1,1),
                                seasonal_order=(0,1,1,4))
    res_sarimax = mod_sarimax.fit()

    #res = mod_sarimax.filter(res_sarimax.params)

    # Show the summary of results
    pred = res_sarimax.get_prediction(window,window).predicted_mean
    
    predict_sarima.append(pred)


# In[70]:


res.get_prediction(18000).predicted_mean


# In[121]:


predict_sarima = np.array(predict_sarima).squeeze()
abs(predict_sarima - xs[window : window+10]).mean()


# In[116]:


abs(xs - pred.astype(int)).mean()


# In[1]:


xs[window: window + 100]
print(predict_sarima)


# In[113]:


predict_sarima = np.array(predict_sarima).astype(int).reshape(1,-1)
print((predict_sarima == xs[window: window + 100]).astype(int).mean())


# In[60]:


get_ipython().run_line_magic('pip', 'install hmmlearn')


# In[69]:



remodel = hmm.CategoricalHMM(n_components=5)
remodel.fit(xs.reshape(-1,1))
Z2 = remodel.predict(xs.reshape(-1,1))


# In[73]:


print(abs(Z2-xs).mean())


# In[ ]:




