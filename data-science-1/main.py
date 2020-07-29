#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[38]:


# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[32]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[71]:


# Sua análise da parte 1 começa aqui.
norm = dataframe.normal.quantile([0.25, 0.5, 0.75])

binom = dataframe.binomial.quantile([0.25, 0.5, 0.75])

print(norm)
print(binom)

print(dataframe['normal'].describe())
print(dataframe['binomial'].describe())

dataframe['normal'].describe().loc['25%']
# df.loc['25%']

q1_norm = norm[0.25]
q2_norm = norm[0.5]
q3_norm = norm[0.75]
q1_binom = binom[0.25]
q2_binom = binom[0.5]
q3_binom = binom[0.75]

print(dataframe['normal'].mean(), dataframe['normal'].var())
    
ans = (q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom)
ans

sct.norm.cdf(24, loc=20, scale=4) - sct.norm.cdf(16, loc=20, scale=4)

m_binom = dataframe['binomial'].mean()
v_binom = dataframe['binomial'].var()
m_norm = dataframe['normal'].mean()
v_norm = dataframe['normal'].var()
m_binom - m_norm, v_binom - v_norm

sct.norm.cdf(24, loc=20, scale=4) - sct.norm.cdf(16, loc=20, scale=4)

normal = dataframe['normal']
m_norm = normal.mean()
s_norm = normal.std()
ecdf = ECDF(normal)
sum = m_norm + s_norm
sub = m_norm - s_norm
calc_sum = ecdf([sum])
calc_sub = ecdf([sub])
res = (calc_sum - calc_sub)
(abs(calc_sum - calc_sub).round(3).item())

type((m_binom - m_norm, v_binom - v_norm))


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[76]:


def q1():
    q1_norm = norm[0.25]
    q2_norm = norm[0.5]
    q3_norm = norm[0.75]
    q1_binom = binom[0.25]
    q2_binom = binom[0.5]
    q3_binom = binom[0.75]
    
    return(tuple((pd.Series([q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom]).round(3))))
    # Retorne aqui o resultado da questão 1.
    pass


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[64]:


def q2():
    normal = dataframe['normal']
    m_norm = normal.mean()
    s_norm = normal.std()
    ecdf = ECDF(normal)
    sum = m_norm + s_norm
    sub = m_norm - s_norm
    calc_sum = ecdf([sum])
    calc_sub = ecdf([sub])
    return(abs(calc_sum - calc_sub).round(3).item())
    # Retorne aqui o resultado da questão 2.
    pass


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[75]:


def q3():
    m_binom = dataframe['binomial'].mean()
    v_binom = dataframe['binomial'].var()
    m_norm = dataframe['normal'].mean()
    v_norm = dataframe['normal'].var()
    
    return(tuple(pd.Series([m_binom - m_norm, v_binom - v_norm]).round(3)))
    # Retorne aqui o resultado da questão 3.
    pass


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[37]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[28]:


# Sua análise da parte 2 começa aqui.

stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[65]:


def q4():
    # Retorne aqui o resultado da questão 4.
    falsas_pulsar = stars[stars.target == 0]
    false_pulsar_mean_profile = falsas_pulsar.mean_profile
    mean = falsas_pulsar.mean_profile.mean()
    std = falsas_pulsar.mean_profile.std()
    false_pulsar_mean_profile_standardized = (falsas_pulsar.mean_profile - mean)/std
    norm_80 = sct.norm.ppf(0.80, loc=0, scale=1)
    norm_90 = sct.norm.ppf(0.90, loc=0, scale=1)
    norm_95 = sct.norm.ppf(0.95, loc=0, scale=1)
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    return(tuple(ecdf([norm_80, norm_90, norm_95]).round(3)))
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[67]:


def q5():
    # Retorne aqui o resultado da questão 5.
    falsas_pulsar = stars[stars.target == 0]
    false_pulsar_mean_profile = falsas_pulsar.mean_profile
    mean = falsas_pulsar.mean_profile.mean()
    std = falsas_pulsar.mean_profile.std()
    false_pulsar_mean_profile_standardized = (falsas_pulsar.mean_profile - mean)/std
    norm_25 = sct.norm.ppf(0.25, loc=0, scale=1)
    norm_50 = sct.norm.ppf(0.50, loc=0, scale=1)
    norm_75 = sct.norm.ppf(0.75, loc=0, scale=1)
    norm_data = np.percentile(false_pulsar_mean_profile_standardized, [25,50,75])
    norm_teor = [norm_25, norm_50, norm_75]
    calc = norm_data - norm_teor
    return(tuple(pd.Series(calc).round(3)))
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
