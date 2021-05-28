from collections import Counter
import csv
import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv("textblob_unclean_csv.csv")

#print(df['user_name'].value_counts())

Days = ['2020-06-15', '2020-06-16','2020-06-17', '2020-06-18', '2020-06-19', '2020-06-20', '2020-06-21']
n = 38583
i = 0
j = 0

u1_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
u1_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
u1_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

u2_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
u2_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
u2_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

u3_pos_count = np.array([0, 0, 0, 0, 0, 0, 0])
u3_neg_count = np.array([0, 0, 0, 0, 0, 0, 0])
u3_neu_count = np.array([0, 0, 0, 0, 0, 0, 0])

for i in range(n):
    if ((df.loc[i, 'user_name'] == 'CNNnews18')) == 1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u1_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u1_neg_count[j] += 1
            else:
                u1_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u1_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u1_neg_count[j+1] += 1
            else:
                u1_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u1_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u1_neg_count[j+2] += 1
            else:
                u1_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u1_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u1_neg_count[j+3] += 1
            else:
                u1_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u1_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u1_neg_count[j+4] += 1
            else:
                u1_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u1_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u1_neg_count[j+5] += 1
            else:
                u1_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u1_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u1_neg_count[j+6] += 1
            else:
                u1_neu_count[j+6] += 1

    if ((df.loc[i, 'user_name'] == 'TimesNow')) == 1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u2_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u2_neg_count[j] += 1
            else:
                u2_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u2_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u2_neg_count[j+1] += 1
            else:
                u2_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u2_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u2_neg_count[j+2] += 1
            else:
                u2_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u2_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u2_neg_count[j+3] += 1
            else:
                u2_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u2_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u2_neg_count[j+4] += 1
            else:
                u2_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u2_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u2_neg_count[j+5] += 1
            else:
                u2_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u2_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u2_neg_count[j+6] += 1
            else:
                u2_neu_count[j+6] += 1

    if ((df.loc[i, 'user_name'] == 'mohitsmartlove')) == 1:
        if df.loc[i, 'date'].find('-15') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u3_pos_count[j] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u3_neg_count[j] += 1
            else:
                u3_neu_count[j] += 1
        if df.loc[i, 'date'].find('-16') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u3_pos_count[j+1] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u3_neg_count[j+1] += 1
            else:
                u3_neu_count[j+1] += 1
        if df.loc[i, 'date'].find('-17') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u3_pos_count[j+2] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u3_neg_count[j+2] += 1
            else:
                u3_neu_count[j+2] += 1
        if df.loc[i, 'date'].find('-18') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u3_pos_count[j+3] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u3_neg_count[j+3] += 1
            else:
                u3_neu_count[j+3] += 1
        if df.loc[i, 'date'].find('-19') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u3_pos_count[j+4] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u3_neg_count[j+4] += 1
            else:
                u3_neu_count[j+4] += 1
        if df.loc[i, 'date'].find('-20') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u3_pos_count[j+5] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u3_neg_count[j+5] += 1
            else:
                u3_neu_count[j+5] += 1
        if df.loc[i, 'date'].find('-21') != -1:
            if df.loc[i, 'TextBlob_Sentiment_clean']==1:
                u3_pos_count[j+6] += 1
            elif df.loc[i, 'TextBlob_Sentiment_clean']==-1:
                u3_neg_count[j+6] += 1
            else:
                u3_neu_count[j+6] += 1

x_indexs = np.arange(len(Days))
BarWidth = 0.25
epsilon = 0.006
opacity = 0.75

pos_posn = np.arange(len(Days))
neg_posn = pos_posn + BarWidth
neu_posn = neg_posn + BarWidth

pos_bar_1 = plt.bar(pos_posn, u1_pos_count, BarWidth-epsilon, color = 'blue', label='Positive', alpha = opacity)
neg_bar_1 = plt.bar(pos_posn, u1_neg_count, BarWidth-epsilon, color = 'red', bottom = u1_pos_count, alpha = opacity, label='Negative')
neu_bar_1 = plt.bar(pos_posn, u1_neu_count, BarWidth-epsilon, color = 'green', bottom = u1_neg_count + u1_pos_count, alpha = opacity, label='Neutral')

pos_bar_2 = plt.bar(neg_posn, u2_pos_count, BarWidth-epsilon, color = 'blue', alpha = opacity)
neg_bar_2 = plt.bar(neg_posn, u2_neg_count, BarWidth-epsilon, color = 'red', bottom = u2_pos_count, alpha = opacity)
neu_bar_2 = plt.bar(neg_posn, u2_neu_count, BarWidth-epsilon, color = 'green', bottom = u2_pos_count + u2_neg_count, alpha = opacity)

pos_bar_3 = plt.bar(neu_posn, u3_pos_count, BarWidth-epsilon, color = 'blue', alpha = opacity)
neg_bar_3 = plt.bar(neu_posn, u3_neg_count, BarWidth-epsilon, color = 'red', bottom = u3_pos_count, alpha = opacity)
neu_bar_3 = plt.bar(neu_posn, u3_neu_count, BarWidth-epsilon, color = 'green', bottom = u3_pos_count + u3_neg_count, alpha = opacity)


text = plt.annotate("CNNnews18", xy=(0,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("TimesNow", xy=(0.25,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("mohitsmartlove", xy=(0.50,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("CNNnews18", xy=(1,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("TimesNow", xy=(1.25,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("mohitsmartlove", xy=(1.50,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("CNNnews18", xy=(2,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("TimesNow", xy=(2.25,60), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("mohitsmartlove", xy=(2.50,60), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("CNNnews18", xy=(3,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("TimesNow", xy=(3.25,18), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("mohitsmartlove", xy=(3.50,20), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("CNNnews18", xy=(4, 60), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("TimesNow", xy=(4.25,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("mohitsmartlove", xy=(4.50,22), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("CNNnews18", xy=(5,53), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("TimesNow", xy=(5.25,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("mohitsmartlove", xy=(5.50,9), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("CNNnews18", xy=(6,19), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("TimesNow", xy=(6.25,3), ha="center", va="bottom")
text.set_rotation(90)

text = plt.annotate("mohitsmartlove", xy=(6.5,3), ha="center", va="bottom")
text.set_rotation(90)

plt.xticks(pos_posn, Days, rotation=45)
plt.title('Sentiment of top 3 active users over the period of 7 days')
plt.xlabel('Date')
plt.ylabel('Counts')
plt.legend()
#plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.8))
plt.show()

"""
print("\nUser 1: CNNnews18")  #191
print(u1_pos_count)
print(u1_neg_count)
print(u1_neu_count)

print("\nUser 2: TimesNow")  #167
print(u2_pos_count)
print(u2_neg_count)
print(u2_neu_count)

print("\nUser 3: mohitsmartlove")  #163
print(u3_pos_count)
print(u3_neg_count)
print(u3_neu_count)
"""