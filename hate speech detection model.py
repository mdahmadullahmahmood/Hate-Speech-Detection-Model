#!/usr/bin/env python
# coding: utf-8

# In[66]:


# import the required libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# In[67]:


# import the required libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
#from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
#tweet_df = pd.read_csv('hateDetection_train.csv')
#tweet_df.head()


# In[68]:


import nltk


# In[69]:


nltk.download('stopwords')


# In[70]:


# import the required libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
#tweet_df = pd.read_csv('hateDetection_train.csv')
#tweet_df.head()


# In[71]:


# import the required libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style'\\\u
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
#tweet_df = pd.read_csv('hateDetection_train.csv')
#tweet_df.head()


# In[72]:


import wordcloud


# In[73]:


pip install wordcloud


# In[74]:


# import the required libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
#tweet_df = pd.read_csv('hateDetection_train.csv')
#tweet_df.head()


# In[75]:


tweet_df = pd.read_csv('train.csv')
tweet_df.head()


# In[76]:


tweet_df.info()


# In[77]:


# printing random tweets 
print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")


# In[78]:


#creating a function to process the data
def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)


# In[79]:


tweet_df.tweet = tweet_df['tweet'].apply(data_processing)


# In[80]:


nltk.download('punkt')


# In[81]:


import nltk
nltk.download('punkt')


# In[82]:


tweet_df.tweet = tweet_df['tweet'].apply(data_processing)


# In[83]:


tweet_df = tweet_df.drop_duplicates('tweet')


# In[84]:


lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    tweet = [lemmarizer.lemmatize(word) for word in data]
    return data


# In[85]:


tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: lemmatizing(x))


# In[86]:


# printing the data to see the effect of preprocessing
print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")


# In[87]:


tweet_df.info()


# In[88]:


tweet_df['label'].value_counts()


# In[89]:


fig = plt.figure(figsize=(5,5))
sns.countplot(x='label', data = tweet_df)


# In[90]:


fig = plt.figure(figsize=(7,7))
colors = ("red", "gold")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = tweet_df['label'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie',autopct = '%1.1f%%', shadow=True, colors = colors, startangle =90, 
         wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')


# In[91]:


non_hate_tweets = tweet_df[tweet_df.label == 0]
non_hate_tweets.head()


# In[92]:


text = ' '.join([word for word in non_hate_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in non hate tweets', fontsize = 19)
plt.show()


# In[93]:


neg_tweets = tweet_df[tweet_df.label == 1]
neg_tweets.head()


# In[94]:


text = ' '.join([word for word in neg_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in hate tweets', fontsize = 19)
plt.show()


# In[95]:


vect = TfidfVectorizer(ngram_range=(1,2)).fit(tweet_df['tweet'])


# In[96]:


feature_names = vect.get_feature_names_out()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features: \n{}".format(feature_names[:20]))


# In[117]:


X = tweet_df['tweet']
Y = tweet_df['label']
X = vect.transform(X)


# In[98]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[99]:


print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))


# In[100]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_predict = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_predict, y_test)
print("Test accuarcy: {:.2f}%".format(logreg_acc*100))


# In[101]:


print(confusion_matrix(y_test, logreg_predict))
print("\n")
print(classification_report(y_test, logreg_predict))


# In[102]:


style.use('classic')
cm = confusion_matrix(y_test, logreg_predict, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()


# In[103]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[104]:


param_grid = {'C':[100, 10, 1.0, 0.1, 0.01], 'solver' :['newton-cg', 'lbfgs','liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
grid.fit(x_train, y_train)
print("Best Cross validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)


# In[105]:


y_pred = grid.predict(x_test)


# In[106]:


logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[107]:


print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))


# In[108]:


sent = 'dogs are cute'


# In[109]:


print(x_test)


# In[110]:


print(X)


# In[111]:


X = vect.transform(['roses are red'])


# In[112]:


y_pred = grid.predict(X)


# In[113]:


print(y_pred)


# In[114]:


X = vect.transform(['all blacks are criminals'])
y_pred = grid.predict(X)
print(y_pred)


# In[116]:


def getmodelprediction(sent):
    arr = []
    arr.append(sent)
    X = vect.transform(arr)
    y_pred = grid.predict(X)
    if y_pred == 0:
        return "No hate speech detected"
    elif y_pred == 1:
        return "Hate Speech detected"


# In[56]:


getmodelprediction('nazis were great')


# In[53]:


getmodelprediction('ronaldo is great player')


# In[54]:


getmodelprediction('hello')


# In[55]:


getmodelprediction('hello world')


# In[ ]:




