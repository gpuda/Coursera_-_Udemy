
# coding: utf-8

# # Exercise 2
# ## Part 1
# Now let's calculate covariance and correlation by ourselves using ApacheSpark
# 
# 1st we crate two random RDD’s, which shouldn't correlate at all.
# 

# In[1]:


import random
rddX = sc.parallelize(random.sample(range(100),100))
rddY = sc.parallelize(random.sample(range(100),100))


# Now we calculate the mean, note that we explicitly cast the denominator to float in order to obtain a float instead of int

# In[2]:


meanX = rddX.sum()/float(rddX.count())
meanY = rddY.sum()/float(rddY.count())
print meanX
print meanY


# Now we calculate the covariance

# In[5]:


rddXY = rddX.zip(rddY)
covXY = rddXY.map(lambda (x,y) : (x-meanX)*(y-meanY)).sum()/rddXY.count()
covXY


# Covariance is not a normalized measure. Therefore we use it to calculate correlation. But before that we need to calculate the indivicual standard deviations first

# In[6]:


from math import sqrt
n = rddXY.count()
sdX = sqrt(rddX.map(lambda x : pow(x-meanX,2)).sum()/n)
sdY = sqrt(rddY.map(lambda x : pow(x-meanY,2)).sum()/n)
print sdX
print sdY


# Now we calculate the correlation

# In[7]:


corrXY = covXY / (sdX * sdY)
corrXY


# ## Part 2
# No we want to create a correlation matrix out of the four RDDs used in the lecture

# In[8]:


from pyspark.mllib.stat import Statistics
import random
column1 = sc.parallelize(range(100))
column2 = sc.parallelize(range(100,200))
column3 = sc.parallelize(list(reversed(range(100))))
column4 = sc.parallelize(random.sample(range(100),100))
data = column1.zip(column2).zip(column3).zip(column4).map(lambda (((a,b),c),d) : (a,b,c,d) ).map(lambda (a,b,c,d) : [a,b,c,d])
print(Statistics.corr(data))


# Congratulations, you are done with Exercice 2
