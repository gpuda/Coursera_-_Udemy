
# coding: utf-8

# In[1]:


from systemml import dml, MLContext
import numpy as np


# In[2]:


ml = MLContext (spark)


# In[10]:


script = """
c = sum(a %*% t(b))
"""


# In[16]:


a = np.array([[1,2,3]])
b = np.array([[4,5,6]])
prog = dml(script).input('a', a).input('b', b).output('c')
c = ml.execute(prog).get('c')
print (c)

