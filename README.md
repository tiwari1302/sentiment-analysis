# Twitter Sentiment-analysis
This repository contains script, files and results of the study conducted in India during lockdown in June 2020 as a part of internship at IIT Hyderabad.

This script can tell you the sentiments of people regarding any events happening in the world by analyzing tweets related to that event.

These scripts were written during COVID-19 wave 1, when the nation was going through lockdown and there was some tension at the Indo-China border. But you can fork and modify the scripts and put your own hashtags to search.

## Getting Started 
First of all, login from your Twitter account and go to Twitter Apps. Create a new app and go to Keys and access tokens and copy Consumer Key, Consumer Secret, Access Token and Access Token Secret. We will need them later.

## Usage
Make sure you create a separate virtual environment for this project. You need some libraries installed in your virtual environment in order to access these scripts. For eg., jupyter, tweepy, pandas, matplotlib, TextBlob and many more. (I will be adding a requirements.txt soon)

Download or clone the repo.
On cmd or bash:
 ```
 cd "sentiment-analysis\code"
 ```
 Run ``` jupyter notebook```
 
 Open dataExtract.py
 Edit the hashtags you want to search for, and then run the script. You will have a csv file ready in your directory in some time. 
 
 The file will contain only the necessary information that is required to perform analysis, like location info, user info, tweet text, its date and time, and some more information.
 
 Then you can open ```preprocessing and cleaning.ipynb``` and make necessary changes to that as well. It includes code to perform preprocessing. Once the data is preprocessed, we have to perform sentiment analysis on that data. 
 
 Once the sentiment analysis is done, the results can be visualised. 
 Run ```cd  plotting```. This directory involves scripts to visualise the results in form of plots.
 
 ## Contributing

1. Fork this repo.
2. Create your feature branch: ```git checkout -b my-new-feature```.
3. Commit your changes: ```git commit -am 'Add some feature'```
4. Push to the branch: ```git push origin my-new-feature```.
5. Submit a pull request.

## Authors
Shashank Tiwari
