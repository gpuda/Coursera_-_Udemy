
# coding: utf-8

# This is the first assgiment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# The first step is to insert the credentials to the Apache CouchDB / Cloudant database where your sensor data ist stored to. 
# 
# 1. In the project's overview tab of this project just select "Add to project"->Connection
# 2. From the section "Your service instances in IBM Cloud" select your cloudant database and click on "create"
# 3. Now click in the empty cell below labeled with "your cloudant credentials go here"
# 4. Click on the "10-01" symbol top right and selecrt the "Connections" tab
# 5. Find your data base connection and click on "Insert to code"
# 
# Done, just execute all cells one after the other and you are done - just note that in the last one you have to update your email address (the one you've used for coursera) and obtain a submittion token, you get this from the programming assingment directly on coursera.

# In[14]:


# @hidden_cell
# @hidden_cell
credentials_2 = {
  'password':"""d787ebc40c3964ed31ee1f2ca2f13c62d088efa0ed5936cd276567c779158215""",
  'custom_url':'https://54e1866b-8a1f-4f2e-bc55-918e9a524c7b-bluemix:d787ebc40c3964ed31ee1f2ca2f13c62d088efa0ed5936cd276567c779158215@54e1866b-8a1f-4f2e-bc55-918e9a524c7b-bluemix.cloudant.com',
  'username':'54e1866b-8a1f-4f2e-bc55-918e9a524c7b-bluemix',
  'url':'https://undefined'
}
#your cloudant credentials go here


# In[15]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_2['custom_url'].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .getOrCreate()


# In[21]:


df=spark.read.load('shake_demo', "org.apache.bahir.cloudant")

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# In[22]:


get_ipython().system(u'rm -Rf a2_m1.parquet')


# In[23]:


df = df.repartition(1)
df.write.json('a2_m1.json')


# In[24]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[25]:


get_ipython().system(u'zip -r a2_m1.json.zip a2_m1.json')


# In[26]:


get_ipython().system(u'base64 a2_m1.json.zip > a2_m1.json.zip.base64')


# In[31]:


from rklib import submit
key = "1injH2F0EeiLlRJ3eJKoXA"
part = "wNLDt"
email = "gpudja@gmail.com"
secret = "pF3RRAbktjF2PMvn"

with open('a2_m1.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)

