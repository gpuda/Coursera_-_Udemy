
# coding: utf-8

# ### Please run the following cell once.
# 
# This will install the Apache Bahir connector within your Project/Apache Spark service. If you restart the kernel or start a new notebook in the same project you can use Apache Bahir for connecting to the Cloudant/Apache CouchDB service. Please note that this will install a patched version of the connector (since the pull request hasn't been merged with the trunk yet).
# 
# You'll find more information on the patch here:
# 
# https://github.com/apache/bahir/pull/49 https://issues.apache.org/jira/browse/BAHIR-130
# 

# In[1]:


import pixiedust
pixiedust.installPackage("https://github.com/romeokienzler/developerWorks/raw/master/coursera/spark-sql-cloudant_2.11-2.3.0-SNAPSHOT.jar")
pixiedust.installPackage("com.typesafe:config:1.3.1")
pixiedust.installPackage("com.typesafe.play:play-json_2.11:jar:2.5.9")
pixiedust.installPackage("org.scalaj:scalaj-http_2.11:jar:2.3.0")
pixiedust.installPackage("com.typesafe.play:play-functional_2.11:jar:2.5.9")

