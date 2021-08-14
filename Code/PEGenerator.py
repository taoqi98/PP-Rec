from keras.utils import Sequence
import numpy as np

FLAG_CTR = 1

def compute_Q_publish(publish_time):
    arg = publish_time<0
    publish_time[arg] = 0
    return publish_time//2

def fetch_ctr_dim1(News,docids,bucket,flag = 1):
    doc_imp = News.news_stat_imp[docids]
    doc_click = News.news_stat_click[docids]
    
    if flag == 1:
        ctr = doc_click[:,bucket-1]/(doc_imp[:,bucket-1]+0.01)
    return ctr

def fetch_ctr_dim2(News,docids,bucket,flag = 1):
    batch_size, doc_num = docids.shape
    #print(docids.shape)
    ctr = np.zeros(docids.shape)
    doc_imp = News.news_stat_imp[docids]
    doc_click = News.news_stat_click[docids]
    #print(doc_click.shape)
    for i in range(batch_size):
        if flag == 1:
            ctr[i,:] = doc_click[i,:,bucket[i]-1]/(doc_imp[i,:,bucket[i]-1]+0.01)
    
    return ctr

def fetch_ctr_dim3(News,docids,bucket,flag = 1):
    batch_size, doc_num = docids.shape
    doc_imp = News.news_stat_imp[docids]
    doc_click = News.news_stat_click[docids]
    ctr = np.zeros(docids.shape)
    for i in range(batch_size):
        for j in range(doc_num):
            b = bucket[i,j]-1
            if b<0:
                b = 0
            ctr[i,j] = doc_click[i,j,b]/(doc_imp[i,j,b]+0.01)
    ctr = ctr*200
    ct = np.ceil(ctr)
    ctr = np.array(ctr,dtype='int32')
    return ctr


class TrainGenerator(Sequence):
    def __init__(self, News,Users, news_id, userids, buckets,label, batch_size):
        
        self.News = News
        self.Users = Users

        self.userids = userids        
        self.doc_id = news_id
        self.buckets = buckets
        self.label = label
        
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        
        doc_ids = self.doc_id[start:ed]
        news_feature = self.News.fetch_news(doc_ids)
        
        userids = self.userids[start:ed]
        clicked_ids = self.Users.click[userids]
        user_feature = self.News.fetch_news(clicked_ids)
        
        bucket = self.buckets[start:ed]
        candidate_ctr = fetch_ctr_dim2(self.News,doc_ids,bucket,FLAG_CTR)
        
        click_bucket = self.Users.click_bucket[userids]
        click_ctr = fetch_ctr_dim3(self.News,clicked_ids,click_bucket,FLAG_CTR)

        bucket = bucket.reshape((bucket.shape[0],1))
        rece = bucket - self.News.news_publish_bucket2[doc_ids]
        
        rece_emb_index = compute_Q_publish(rece)
        user_activity = (clicked_ids>0).sum(axis=-1)
        
        label = self.label[start:ed]
                
        return ([news_feature,candidate_ctr,rece_emb_index,user_activity,user_feature,click_ctr,],[label])

class UserGenerator(Sequence):
    def __init__(self,News, Users,batch_size):
        
        self.News = News
        self.Users = Users

        self.batch_size = batch_size
        self.ImpNum = self.Users.click.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
            
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum

        clicked_ids = self.Users.click[start:ed]
        user_feature = self.News.fetch_news(clicked_ids)
        click_bucket = self.Users.click_bucket[start:ed]
        click_ctr = fetch_ctr_dim3(self.News,clicked_ids,click_bucket,FLAG_CTR)


        return [user_feature,click_ctr]

class NewsGenerator(Sequence):
    def __init__(self,News,batch_size):
        self.News = News

        self.batch_size = batch_size        
        self.ImpNum = self.News.title.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        docids = np.array([i for i in range(start,ed)])
        
        news_feature = self.News.fetch_news(docids)
        
        return news_feature





