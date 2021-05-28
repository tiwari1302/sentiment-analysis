from collections import Counter
import csv
import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv("../Data/full_csv_with_hashtags.csv")


Days = ['2020-06-15', '2020-06-16','2020-06-17', '2020-06-18', '2020-06-19', '2020-06-20', '2020-06-21']
n = 38583
i = 0
j = 0

h1_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
h1_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
h1_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

h2_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
h2_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
h2_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

h3_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
h3_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
h3_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

for i in range(n):
    if (df.loc[i, 'tweet'].find('#IndianArmy')) != -1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h1_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h1_neg_count[j] += 1
            else:
                h1_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h1_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h1_neg_count[j+1] += 1
            else:
                h1_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h1_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h1_neg_count[j+2] += 1
            else:
                h1_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h1_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h1_neg_count[j+3] += 1
            else:
                h1_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h1_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h1_neg_count[j+4] += 1
            else:
                h1_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h1_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h1_neg_count[j+5] += 1
            else:
                h1_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h1_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h1_neg_count[j+6] += 1
            else:
                h1_neu_count[j+6] += 1
    
    if (df.loc[i, 'tweet'].find('#GalwanValley')) != -1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h2_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h2_neg_count[j] += 1
            else:
                h2_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h2_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h2_neg_count[j+1] += 1
            else:
                h2_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h2_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h2_neg_count[j+2] += 1
            else:
                h2_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h2_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h2_neg_count[j+3] += 1
            else:
                h2_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h2_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h2_neg_count[j+4] += 1
            else:
                h2_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h2_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h2_neg_count[j+5] += 1
            else:
                h2_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h2_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h2_neg_count[j+6] += 1
            else:
                h2_neu_count[j+6] += 1

    if (df.loc[i, 'tweet'].find('#China')) != -1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h3_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h3_neg_count[j] += 1
            else:
                h3_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h3_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h3_neg_count[j+1] += 1
            else:
                h3_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h3_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h3_neg_count[j+2] += 1
            else:
                h3_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h3_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h3_neg_count[j+3] += 1
            else:
                h3_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h3_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h3_neg_count[j+4] += 1
            else:
                h3_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h3_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h3_neg_count[j+5] += 1
            else:
                h3_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                h3_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                h3_neg_count[j+6] += 1
            else:
                h3_neu_count[j+6] += 1
    
x_indexs = np.arange(len(Days))
BarWidth = 0.25
epsilon = 0.006
opacity = 0.75

pos_posn = np.arange(len(Days))
neg_posn = pos_posn + BarWidth
neu_posn = neg_posn + BarWidth

pos_bar_d = plt.bar(pos_posn, h2_pos_count, BarWidth-epsilon, color = 'blue', label='Positive', alpha = opacity)
neg_bar_d = plt.bar(pos_posn, h2_neg_count, BarWidth-epsilon, color = 'red', bottom = h2_pos_count, alpha = opacity, label='Negative')
neu_bar_d = plt.bar(pos_posn, h2_neu_count, BarWidth-epsilon, color = 'green', bottom = h2_neg_count + h2_pos_count, alpha = opacity, label='Neutral')

pos_bar_m = plt.bar(neg_posn, h1_pos_count, BarWidth-epsilon, color = 'blue', alpha = opacity)
neg_bar_m = plt.bar(neg_posn, h1_neg_count, BarWidth-epsilon, color = 'red', bottom = h1_pos_count, alpha = opacity)
neu_bar_m = plt.bar(neg_posn, h1_neu_count, BarWidth-epsilon, color = 'green', bottom = h1_pos_count + h1_neg_count, alpha = opacity)

pos_bar_h = plt.bar(neu_posn, h3_pos_count, BarWidth-epsilon, color = 'blue', alpha = opacity)
neg_bar_h = plt.bar(neu_posn, h3_neg_count, BarWidth-epsilon, color = 'red', bottom = h3_pos_count, alpha = opacity)
neu_bar_h = plt.bar(neu_posn, h3_neu_count, BarWidth-epsilon, color = 'green', bottom = h3_pos_count + h3_neg_count, alpha = opacity)

text = plt.annotate("#IndianArmy", xy=(0,14), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#GalwanValley", xy=(0.25,8), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#China", xy=(0.50,30), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#IndianArmy", xy=(1,403), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#GalwanValley", xy=(1.25,403), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#China", xy=(1.50,403), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#IndianArmy", xy=(2,200), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#GalwanValley", xy=(2.25,370), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#China", xy=(2.50,330), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#IndianArmy", xy=(3,120), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#GalwanValley", xy=(3.25,260), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#China", xy=(3.50,190), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#IndianArmy", xy=(4, 350), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#GalwanValley", xy=(4.25,358), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#China", xy=(4.50,540), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#IndianArmy", xy=(5,582), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#GalwanValley", xy=(5.25,200), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#China", xy=(5.50,310), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#IndianArmy", xy=(6,150), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#GalwanValley", xy=(6.25,115), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("#China", xy=(6.5,267), ha="center", va="bottom")
text.set_rotation(90)

plt.xticks(pos_posn, Days, rotation=45)
plt.title('Sentiment of top 3 hashtags over the period of 7 days')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
#plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.8))
# plt.annotate(Days, ('Delhi', '#GalwanValley', '#China'))
plt.show()

"""
print("\n#IndianArmy Sentiments over the period of a week")
print("Positive", h2_pos_count)
print("Negative", h2_neg_count)
print("Neutral", h2_neu_count)

print("#GalwanValley Sentiments over the period of a week")
print("Positive", h1_pos_count)
print("Negative", h1_neg_count)
print("Neutral", h1_neu_count)

print("\#China Sentiments over the period of a week")
print("Positive", h3_pos_count)
print("Negative", h3_neg_count)
print("Neutral", h3_neu_count)
"""
