
# coding: utf-8

# This is the second assignment for the Coursera course "Advanced Machine Learning and Signal Processing"
# 
# Again, please insert to code to your ApacheCouchDB based Cloudant instance below using the "Insert Code" function of Watson Studio( you've done this in Assignment 1 before)
# 
# Done, just execute all cells one after the other and you are done - just note that in the last one you have to update your email address (the one you've used for coursera) and obtain a submission token, you get this from the programming assignment directly on coursera.
# 
# Please fill in the sections labelled with "###YOUR_CODE_GOES_HERE###"

# In[1]:


#your cloudant credentials go here
###YOUR_CODE_GOES_HERE###"
# @hidden_cell
credentials_1 = {
  'password':"""d787ebc40c3964ed31ee1f2ca2f13c62d088efa0ed5936cd276567c779158215""",
  'custom_url':'https://54e1866b-8a1f-4f2e-bc55-918e9a524c7b-bluemix:d787ebc40c3964ed31ee1f2ca2f13c62d088efa0ed5936cd276567c779158215@54e1866b-8a1f-4f2e-bc55-918e9a524c7b-bluemix.cloudant.com',
  'username':'54e1866b-8a1f-4f2e-bc55-918e9a524c7b-bluemix',
  'url':'https://undefined'
}


# Let's create a SparkSession object and put the Cloudant credentials into it

# In[2]:


spark = SparkSession    .builder    .appName("Cloudant Spark SQL Example in Python using temp tables")    .config("cloudant.host",credentials_1['custom_url'].split('@')[1])    .config("cloudant.username", credentials_1['username'])    .config("cloudant.password",credentials_1['password'])    .config("jsonstore.rdd.partitions", 1)    .getOrCreate()


# Now it’s time to have a look at the recorded sensor data. You should see data similar to the one exemplified below….
# 

# In[5]:


df=spark.read.load('shake_classification', "org.apache.bahir.cloudant")

df.createOrReplaceTempView("df")
spark.sql("SELECT * from df").show()


# Please create a VectorAssembler which consumed columns X, Y and Z and produces a column “features”
# 

# In[12]:


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler (inputCols=["X","Y","Z"],
                                  outputCol="features")


# ##### Please insatiate a classifier from the SparkML package and assign it to the classifier variable. Make sure to either
# 1.	Rename the “CLASS” column to “label” or
# 2.	Specify the label-column correctly to be “CLASS”
# 

# In[18]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder

indexer = StringIndexer (inputCol = "CLASS", outputCol = "label")


# In[19]:


from pyspark.ml.classification import GBTClassifier

classifier = GBTClassifier (labelCol = "label", featuresCol = "features", maxIter=10)



# Let’s train and evaluate…
# 

# In[20]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, indexer, classifier])


# In[21]:


model = pipeline.fit(df)


# In[23]:


prediction = model.transform(df)


# In[24]:


prediction.show()


# In[25]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
binEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("label")
    
binEval.evaluate(prediction) 


# If you are happy with the result (I’m happy with > 0.55) please submit your solution to the grader by executing the following cells, please don’t forget to obtain an assignment submission token (secret) from the Courera’s graders web page and paste it to the “secret” variable below, including your email address you’ve used for Coursera. (0.55 means that you are performing better than random guesses)
# 

# In[26]:


get_ipython().system(u'rm -Rf a2_m2.json')


# In[27]:


prediction = prediction.repartition(1)
prediction.write.json('a2_m2.json')


# In[28]:


get_ipython().system(u'rm -f rklib.py')
get_ipython().system(u'wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[29]:


get_ipython().system(u'zip -r a2_m2.json.zip a2_m2.json')


# In[30]:


get_ipython().system(u'base64 a2_m2.json.zip > a2_m2.json.zip.base64')


# In[33]:


from rklib import submit
key = "J3sDL2J8EeiaXhILFWw2-g"
part = "G4P6f"
email = "gpudja@gmail.com"
secret = "sWSslVyXiDusVl37"

with open('a2_m2.json.zip.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)

