
# coding: utf-8

# #Predicting sentiment from product reviews
# 
# #Fire up GraphLab Create

# In[3]:

import graphlab


# #Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[4]:

products = graphlab.SFrame('amazon_baby.gl/')


# #Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[5]:

products.head()


# #Build the word count vector for each review

# In[6]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[7]:

products.head()


# In[8]:

graphlab.canvas.set_target('ipynb')


# In[10]:

products['name'].show()


# #Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[11]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[12]:

len(giraffe_reviews)


# In[13]:

giraffe_reviews['rating'].show(view='Categorical')


# #Build a sentiment classifier

# In[14]:

products['rating'].show(view='Categorical')


# ##Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[15]:

#ignore all 3* reviews
products = products[products['rating'] != 3]


# In[16]:

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[18]:

products.head()


# ##Let's train the sentiment classifier

# In[19]:

train_data,test_data = products.random_split(.8, seed=0)


# In[20]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# #Evaluate the sentiment model

# In[21]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[22]:

sentiment_model.show(view='Evaluation')


# #Applying the learned model to understand sentiment for Giraffe

# In[23]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[21]:

giraffe_reviews.head()


# ##Sort the reviews based on the predicted sentiment and explore

# In[24]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[25]:

giraffe_reviews.head()


# ##Most positive reviews for the giraffe

# In[26]:

giraffe_reviews[0]['review']


# In[27]:

giraffe_reviews[1]['review']


# ##Show most negative reviews for giraffe

# In[28]:

giraffe_reviews[-1]['review']


# In[29]:

giraffe_reviews[-2]['review']


# # ASSIGNMENT

# In[30]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[33]:

def awesome_count(word_count):
    if 'awesome' in word_count:
        return word_count['awesome']
    else:
        return 0
products['awesome'] = products['word_count'].apply(awesome_count)

def great_count(word_count):
    if 'great' in word_count:
        return word_count['great']
    else:
        return 0
products['great'] = products['word_count'].apply(great_count)

def fantastic_count(word_count):
    if 'fantastic' in word_count:
        return word_count['fantastic']
    else:
        return 0
products['fantastic'] = products['word_count'].apply(fantastic_count)

def amazing_count(word_count):
    if 'amazing' in word_count:
        return word_count['amazing']
    else:
        return 0
products['amazing'] = products['word_count'].apply(amazing_count)

def love_count(word_count):
    if 'love' in word_count:
        return word_count['love']
    else:
        return 0
products['love'] = products['word_count'].apply(love_count)

def horrible_count(word_count):
    if 'horrible' in word_count:
        return word_count['horrible']
    else:
        return 0
products['horrible'] = products['word_count'].apply(horrible_count)

def bad_count(word_count):
    if 'bad' in word_count:
        return word_count['bad']
    else:
        return 0
products['bad'] = products['word_count'].apply(bad_count)

def terrible_count(word_count):
    if 'terrible' in word_count:
        return word_count['terrible']
    else:
        return 0
products['terrible'] = products['word_count'].apply(terrible_count)

def awful_count(word_count):
    if 'awful' in word_count:
        return word_count['awful']
    else:
        return 0
products['awful'] = products['word_count'].apply(awful_count)

def wow_count(word_count):
    if 'wow' in word_count:
        return word_count['wow']
    else:
        return 0
products['wow'] = products['word_count'].apply(wow_count)

def hate_count(word_count):
    if 'hate' in word_count:
        return word_count['hate']
    else:
        return 0
products['hate'] = products['word_count'].apply(hate_count)


# In[34]:

products


# In[35]:

for word in selected_words:
    print word, ':' , products[word].sum()


# # CREATING A NEW MODEL

# In[36]:

train_data,test_data = products.random_split(.8, seed=0)


# In[39]:

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                          target = 'sentiment',
                                                          features = selected_words,
                                                          validation_set = test_data)


# # Finding the most positive and negative coefficient 

# In[41]:

coef = selected_words_model['coefficients']
coef.sort('value', ascending = False)


# In[43]:

coef.sort('value', ascending = True)


# # Comparing the accuracy of different sentiment analysis model

# In[44]:

selected_words_model.evaluate(test_data)


# In[45]:

sentiment_model.evaluate(test_data)


# # Interpreting the difference in performance between the models

# In[48]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ'] 


# In[50]:

diaper_champ_reviews.head()


# In[52]:

diaper_champ_reviews['prediction'] = selected_words_model.predict(diaper_champ_reviews, output_type='probability')


# In[54]:

diaper_champ_reviews = diaper_champ_reviews.sort('prediction', ascending = False)


# In[55]:

diaper_champ_reviews.head()


# In[56]:

diaper_champ_reviews[1]['review']


# In[57]:

diaper_champ_reviews[-1]['review']


# In[ ]:



