
# coding: utf-8

# # Assignment 3
# 
# Welcome to Assignment 3. This will be even more fun. Now we will calculate statistical measures on the test data you have created.
# 
# YOU ARE NOT ALLOWED TO USE ANY OTHER 3RD PARTY LIBRARIES LIKE PANDAS. PLEASE ONLY MODIFY CONTENT INSIDE THE FUNCTION SKELETONS
# Please read why: https://www.coursera.org/learn/exploring-visualizing-iot-data/discussions/weeks/3/threads/skjCbNgeEeapeQ5W6suLkA
# . Just make sure you hit the play button on each cell from top to down. There are seven functions you have to implement. Please also make sure than on each change on a function you hit the play button again on the corresponding cell to make it available to the rest of this notebook.
# Please also make sure to only implement the function bodies and DON'T add any additional code outside functions since this might confuse the autograder.
# 
# So the function below is used to make it easy for you to create a data frame from a cloudant data frame using the so called "DataSource" which is some sort of a plugin which allows ApacheSpark to use different data sources.
# 

# In[30]:


#Please don't modify this function
def readDataFrameFromCloudant(database):
    cloudantdata=spark.read.load(database, "org.apache.bahir.cloudant")

    cloudantdata.createOrReplaceTempView("washing")
    spark.sql("SELECT * from washing").show()
    return cloudantdata


# All functions can be implemented using DataFrames, ApacheSparkSQL or RDDs. We are only interested in the result. You are given the reference to the data frame in the "df" parameter and in case you want to use SQL just use the "spark" parameter which is a reference to the global SparkSession object. Finally if you want to use RDDs just use "df.rdd" for obtaining a reference to the underlying RDD object. 
# 
# Let's start with the first function. Please calculate the minimal temperature for the test data set you have created. We've provided a little skeleton for you in case you want to use SQL. You can use this skeleton for all subsequent functions. Everything can be implemented using SQL only if you like.

# In[31]:


def minTemperature(df,spark):
    minvalue=df.agg({"temperature":"min"}).collect()[0]
    mintemp = minvalue["min(temperature)"]
    return mintemp


# Please now do the same for the mean of the temperature

# In[33]:


def meanTemperature(df,spark):
    avgvalue=df.agg({"temperature":"avg"}).collect()[0]
    avgtemp=avgvalue ["avg(temperature)"]
    return avgtemp


# Please now do the same for the maximum of the temperature

# In[34]:


def maxTemperature(df,spark):
    maxvalue=df.agg({"temperature":"max"}).collect()[0]
    maxtemp=maxvalue ["max(temperature)"]
    return maxtemp


# Please now do the same for the standard deviation of the temperature

# In[69]:


def sdTemperature(df,spark):
    rddtempvalue = df.select("temperature").rdd
    rddtemp = rddtempvalue.map(lambda x : x["temperature"])
    temp = rddtemp.filter (lambda x: x is not None).filter(lambda x: x != "")
    n = temp.count()
    sum = temp.sum()
    mean = sum/n
    from math import sqrt
    sd = sqrt(temp.map(lambda x : pow(x-mean,2)).sum()/n)
    return sd##INSERT YOUR CODE HERE##


# Please now do the same for the skew of the temperature. Since the SQL statement for this is a bit more complicated we've provided a skeleton for you. You have to insert custom code at four position in order to make the function work. Alternatively you can also remove everything and implement if on your own. Note that we are making use of two previously defined functions, so please make sure they are correct. Also note that we are making use of python's string formatting capabilitis where the results of the two function calls to "meanTemperature" and "sdTemperature" are inserted at the "%s" symbols in the SQL string.

# In[37]:


def skewTemperature(df,spark):    
    rddtempvalue = df.select("temperature").rdd
    rddtemp = rddtempvalue.map(lambda x : x["temperature"])
    temp = rddtemp.filter (lambda x: x is not None).filter(lambda x: x != "")
    n = temp.count()
    sum = temp.sum()
    mean = sum/n
    from math import sqrt
    sd = sqrt(temp.map(lambda x : pow(x-mean,2)).sum()/n)
    skewness = 1/n * temp.map(lambda x : pow(x-mean,3)/pow(sd,3)).sum()
    return skewness##INSERT YOUR CODE HERE##


# Kurtosis is the 4th statistical moment, so if you are smart you can make use of the code for skew which is the 3rd statistical moment. Actually only two things are different.

# In[62]:


def kurtosisTemperature(df,spark):  
    rddtempvalue = df.select("temperature").rdd
    rddtemp = rddtempvalue.map(lambda x : x["temperature"])
    temp = rddtemp.filter (lambda x: x is not None).filter(lambda x: x != "")
    n = temp.count()
    sum = temp.sum()
    mean = sum/n
    from math import sqrt
    sd = sqrt(temp.map(lambda x : pow(x-mean,2)).sum()/n)
    kurtosis = temp.map(lambda x : pow(x-mean,4)).sum()/(pow(sd,4)*n)
    return kurtosis##INSERT YOUR CODE HERE##


# Just a hint. This can be solved easily using SQL as well, but as shown in the lecture also using RDDs.

# In[67]:


def correlationTemperatureHardness(df,spark):
    rddtempvalue = df.select("temperature").rdd
    rddtemp = rddtempvalue.map(lambda x : x["temperature"])
    temp = rddtemp.filter (lambda x: x is not None).filter(lambda x: x != "")
    rddhardvalue = df.select("hardness").rdd
    rddhard = rddhardvalue.map(lambda x : x["hardness"])
    hard = rddhard.filter (lambda x: x is not None).filter(lambda x: x != "")
    ntemp = temp.count()
    #nhard = hard.count()
    sum_temp = temp.sum()
    sum_hard = hard.sum()
    meantemp = sum_temp/ntemp
    meanhard = sum_hard/ntemp
    rdd_t_h = temp.zip(hard)
    cov_t_h = rdd_t_h.map(lambda (x,y) : (x - meantemp)*(y - meanhard)).sum()/ntemp
    from math import sqrt
    sdtemp = sqrt(temp.map(lambda x : pow(x-meantemp,2)).sum()/ntemp)
    sdhard = sqrt(hard.map(lambda x : pow(x-meanhard,2)).sum()/ntemp)
    corr_temp_hard = cov_t_h / (sdtemp * sdhard)
    return corr_temp_hard##INSERT YOUR CODE HERE##


# ### PLEASE DON'T REMOVE THIS BLOCK - THE FOLLOWING CODE IS NOT GRADED
# #axx
# ### PLEASE DON'T REMOVE THIS BLOCK - THE FOLLOWING CODE IS NOT GRADED

# Now it is time to connect to the cloudant database. Please have a look at the Video "Overview of end-to-end scenario" of Week 2 starting from 6:40 in order to learn how to obtain the credentials for the database. Please paste this credentials as strings into the below code
# 
# ### TODO Please provide your Cloudant credentials here

# In[40]:


hostname = "dc5a4654-cfd1-4ed6-92a9-ce5c62c07a9e-bluemix.cloudant.com"
user = "dc5a4654-cfd1-4ed6-92a9-ce5c62c07a9e-bluemix"
pw = "e26a7317318ce637fa77c9bdbe3a6df3522857b38eed9e78d29c4790fac05dd3"
database = "washing" #as long as you didn't change this in the NodeRED flow the database name stays the same


# In[41]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",hostname)    .config("cloudant.username", user)    .config("cloudant.password",pw)    .getOrCreate()
cloudantdata=readDataFrameFromCloudant(database)


# In[42]:


minTemperature(cloudantdata,spark)


# In[43]:


meanTemperature(cloudantdata,spark)


# In[44]:


maxTemperature(cloudantdata,spark)


# In[70]:


sdTemperature(cloudantdata,spark)


# In[46]:


skewTemperature(cloudantdata,spark)


# In[63]:


kurtosisTemperature(cloudantdata,spark)


# In[68]:


correlationTemperatureHardness(cloudantdata,spark)


# Congratulations, you are done, please download this notebook as python file using the export function and submit is to the gader using the filename "assignment3.1.py"
