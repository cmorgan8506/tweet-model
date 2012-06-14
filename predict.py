from numpy import *
import pickle
from sklearn.svm import LinearSVC
from pymongo import Connection as pyconnection

class Predict:
  def __init__(self):
    connection = pyconnection()
    self.db = connection.tweet_leads
    
  def is_relevant(self, tweet):
    model = self.get_model(tweet['campaign_id'])
    if not model:
      datasets = self.get_data(tweet['campaign_id'])
      model = self.build_model(datasets)
      self.save_model(model, tweet['campaign_id'])
    features = self.tweet_features(tweet)
    return model.predict(features)
      
  def get_data(self, campaign_id):
    datasets = self.db.datasets.find_one({"campaign_id": str(campaign_id)})
    return datasets
  
  def build_model(self, dataset):
    target_data = []
    for data in dataset['archive_features']:
      target_data.append(1)
    for data in dataset['logged_features']:
      target_data.append(0)
    
    data = array(dataset['archive_features'] + dataset['logged_features'])
    target = array(target_data)
    model = LinearSVC()
    model = model.fit(data, target)
    return model
    
  def get_model(self, campaign_id):
    model = self.db.models.find_one({'campaign_id': campaign_id})
    model = model['model']
    if model:
      model = pickle.loads(model)
    return model
    
  def save_model(self, model, campaign_id):
    model = pickle.dumps(model)
    self.db.models.insert({'campaign_id': campaign_id, "model": model})

  def get_tweet(self):
    tweet = self.db.archives.find_one()
    return tweet
    
  def tweet_features(self, tweet):
    features = []
    features.append(str(tweet['user']['followers_count']))
    features.append(str(tweet['user']['friends_count']))
    features.append(str(tweet['entities']['hashtags'].__len__()))
    features.append(str(tweet['entities']['user_mentions'].__len__()))
    retweet = str(tweet['retweeted'])
    features.append(str(tweet['entities']['urls'].__len__()))
    features.append(str(tweet['user']['listed_count']))
    features.append(str(tweet['user']['statuses_count']))
    features.append(str(tweet['user']['favourites_count']))
    if retweet:
      features.append('1')
    else:
      features.append('0')
    return features
