from collections import Counter
import csv
import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv("textblob_unclean_csv.csv")

#print(df['loaction'].value_counts())

Days = ['2020-06-15', '2020-06-16','2020-06-17', '2020-06-18', '2020-06-19', '2020-06-20', '2020-06-21']
n = 38583
i = 0
j = 0

d_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
d_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
d_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

m_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
m_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
m_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

h_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
h_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
h_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

for i in range(n):
    if (df.loc[i, 'loaction'].find('Mumbai')) != -1 or (df.loc[i, 'loaction'].find('mumbai')) != -1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                m_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                m_neg_count[j] += 1
            else:
                m_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                m_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                m_neg_count[j+1] += 1
            else:
                m_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                m_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                m_neg_count[j+2] += 1
            else:
                m_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                m_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                m_neg_count[j+3] += 1
            else:
                m_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                m_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                m_neg_count[j+4] += 1
            else:
                m_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                m_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                m_neg_count[j+5] += 1
            else:
                m_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                m_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                m_neg_count[j+6] += 1
            else:
                m_neu_count[j+6] += 1
    
    if (df.loc[i, 'loaction'].find('Delhi')) != -1 or (df.loc[i, 'loaction'].find('delhi')) != -1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                d_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                d_neg_count[j] += 1
            else:
                d_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                d_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                d_neg_count[j+1] += 1
            else:
                d_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                d_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                d_neg_count[j+2] += 1
            else:
                d_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                d_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                d_neg_count[j+3] += 1
            else:
                d_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                d_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                d_neg_count[j+4] += 1
            else:
                d_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                d_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                d_neg_count[j+5] += 1
            else:
                d_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                d_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                d_neg_count[j+6] += 1
            else:
                d_neu_count[j+6] += 1

    if (df.loc[i, 'loaction'].find('Hyderabad')) != -1 or (df.loc[i, 'loaction'].find('hyderabad')) != -1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h_neg_count[j] += 1
            else:
                h_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h_neg_count[j+1] += 1
            else:
                h_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h_neg_count[j+2] += 1
            else:
                h_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h_neg_count[j+3] += 1
            else:
                h_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h_neg_count[j+4] += 1
            else:
                h_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h_neg_count[j+5] += 1
            else:
                h_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h_neg_count[j+6] += 1
            else:
                h_neu_count[j+6] += 1
    
x_indexs = np.arange(len(Days))
BarWidth = 0.25
epsilon = 0.006
opacity = 0.75

pos_posn = np.arange(len(Days))
neg_posn = pos_posn + BarWidth
neu_posn = neg_posn + BarWidth

pos_bar_d = plt.bar(pos_posn, d_pos_count, BarWidth-epsilon, color = 'blue', label='Positive', alpha = opacity)
neg_bar_d = plt.bar(pos_posn, d_neg_count, BarWidth-epsilon, color = 'red', bottom = d_pos_count, alpha = opacity, label='Negative')
neu_bar_d = plt.bar(pos_posn, d_neu_count, BarWidth-epsilon, color = 'green', bottom = d_neg_count + d_pos_count, alpha = opacity, label='Neutral')

pos_bar_m = plt.bar(neg_posn, m_pos_count, BarWidth-epsilon, color = 'blue', alpha = opacity)
neg_bar_m = plt.bar(neg_posn, m_neg_count, BarWidth-epsilon, color = 'red', bottom = m_pos_count, alpha = opacity)
neu_bar_m = plt.bar(neg_posn, m_neu_count, BarWidth-epsilon, color = 'green', bottom = m_pos_count + m_neg_count, alpha = opacity)

pos_bar_h = plt.bar(neu_posn, h_pos_count, BarWidth-epsilon, color = 'blue', alpha = opacity)
neg_bar_h = plt.bar(neu_posn, h_neg_count, BarWidth-epsilon, color = 'red', bottom = h_pos_count, alpha = opacity)
neu_bar_h = plt.bar(neu_posn, h_neu_count, BarWidth-epsilon, color = 'green', bottom = h_pos_count + h_neg_count, alpha = opacity)

text = plt.annotate("New Delhi", xy=(0,14), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Mumbai", xy=(0.25,8), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Hyderabad", xy=(0.50,6), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("New Delhi", xy=(1,725), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Mumbai", xy=(1.25,413), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Hyderabad", xy=(1.50,322), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("New Delhi", xy=(2,740), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Mumbai", xy=(2.25,448), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Hyderabad", xy=(2.50,227), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("New Delhi", xy=(3,448), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Mumbai", xy=(3.25,290), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Hyderabad", xy=(3.50,123), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("New Delhi", xy=(4, 906), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Mumbai", xy=(4.25,460), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Hyderabad", xy=(4.50,312), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("New Delhi", xy=(5,806), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Mumbai", xy=(5.25,502), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Hyderabad", xy=(5.50,234), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("New Delhi", xy=(6,456), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Mumbai", xy=(6.25,267), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("Hyderabad", xy=(6.5,100), ha="center", va="bottom")
text.set_rotation(90)

plt.xticks(pos_posn, Days, rotation=45)
plt.title('Sentiment of top 3 active regions over the period of 7 days')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
#plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.8))
# plt.annotate(Days, ('Delhi', 'Mumbai', 'Hyderabad'))
plt.show()

"""
print("\nDelhi Sentiments over the period of a week")
print("Positive", d_pos_count)
print("Negative", d_neg_count)
print("Neutral", d_neu_count)

print("Mumbai Sentiments over the period of a week")
print("Positive", m_pos_count)
print("Negative", m_neg_count)
print("Neutral", m_neu_count)

print("\Hyderabad Sentiments over the period of a week")
print("Positive", h_pos_count)
print("Negative", h_neg_count)
print("Neutral", h_neu_count)
"""
