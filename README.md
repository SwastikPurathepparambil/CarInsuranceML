# CarInsuranceML

## Introduction

Who doesn't love good ol' car insurance. I am possibly embarking on a journey of driving golf carts around my college campus in a few weeks, and I need insurance on them. This project will help me get acclimated with auto insurance data. And this project will help you (the reader) in no way, shape, or form. Here's a joke I'm sure the Facebook Moms will find comical though (for legal reasons this is a joke). 

Jim´s barn burned down. Julie, his wife, called the insurance company and said, "We had that barn insured for fifty thousand and I want my money."
"Whoa there, just a minute, Julie, it doesn´t work like that. We will assess the value of the building and provide you with a new one of comparable worth." the agent replied.
Julie, after a pause, said, "Well, in that case, I´d like to cancel the policy on my husband."

I'm dying of laughter. Ha ha. Okay, now to the actual project. I will be recording all of my failures as I do this project so I can laugh at my own past mistakes and be amazed at how far I've come. 

## My Choice of ML Model
I decided to use XGBoost as my ML tool of choice because I'm working with lots of tabular data (~58000 rows just for the training set). I worry that my computer will explode though, so I'll first limit the size of my training set to anywhere from 200-5000 rows and see how long it takes to train that. Who knows, maybe the M1 Chip can cook XGBoost no problem. Btw, XGBoost (eXtreme Gradient Boosting) is an ML library that uses boosted decision trees to solve classification and regression problems. It's fast, efficient, and scaleable to large datasets.

## Attempt 1: Failure

XGBClassifier() does not work well with the simple import csv file. For my second attempt, I will work with the pandas dataframe setup for reading csv files.

## Attempt 2: Failure

Didn't add category encoders, my virutal environment really doesn't like me. Category encoders allow things like XGBClassifier() to look at categorical data in a numerical way so that it's easier to process. Cool statistics stuff. 

## Attempt 3: We're getting somewhere?
I pretty much used a YouTube videos code here. Credit goes to the liannewriting GitHub account and Lianne and Justin YouTube account. There's a lot of things I don't really understand here. While this is fairly common when I'm learning something new, I have a split second thought of my life as a farmer in rural Malayasia, digging holes and planting trees for the well being of my family. Anyways, the accuracy of the model is ~63-66%. I'm not even sure if that's bad, so for this next attempt, I will try to understand all of the code I wrote, figure out how to tweak it to work for my dataset, and go from there.

## Attempt 4: Might need to pivot 
I read a bunch of documentation and learned about what each of these things do. Learned that precision and recall are the best ways to determine whether a imbalanced dataset is doing well. The recall and precision values were extremely underwhelming, reaching about 65% for precision and 10% for recall. Maybe the model is underfitting??? Dunno, will figure out how to do optimal hyperparameter tuning.

## Attempt 5: Success
"The today is difficult. Tomorrow is much more difficult. The day after tomorrow is beautiful." - Sun Tzu, The Art of War (I'm pretty sure Jack Ma said this actually). The model works very well now. The f1-score is around 94.5%, which is comprised of a precision of ~97.5% and recall of ~90.5%. This issue was mainly in doing proper hyperparameter tuning and adding all the right hyperparameters into the mix. Also using SMOTE to balance the dataset helped significantly improve the accuracy of the model. After using Bayesian optimization properly and letting the model optimize the hyperparameters (which took about a whole hour to run), everything seems to be working.  