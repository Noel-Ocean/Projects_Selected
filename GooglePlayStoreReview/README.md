### Above the project: 
- Using two datasets of users' reviews of Google Play Store apps
- Factor analysis: Invesigating the factors impacting users' sentiments towards applications
- Sentiment analysis and prediction: Building a classifier to predict users' sentiments (pos/neg)

### About the dataset(s): 
- Source: https://www.kaggle.com/datasets/lava18/google-play-store-apps
- Description: This dataset consists of web scraped data of more than 10,000 Google Play Store apps and 60,000 app reviews. `data_apps.csv` consists of data about the apps such as category, number of installs, and price. `data_reviews.csv` holds reviews of the apps, including the text of the review and sentiment scores.

### Data Dictionary:

**data_apps.csv**

| variable       | class     | description                                                                  |
|:---------------|:----------|:-----------------------------------------------------------------------------|
| App            | character | The application name                                                         |
| Category       | character | The category the app belongs to                                              |
| Rating         | numeric   | Overall user rating of the app                                               |
| Reviews        | numeric   | Number of user reviews for the app                                           |
| Size           | character | The size of the app                                                          |
| Installs       | character | Number of user installs for the app                                          |
| Type           | character | Either "Paid" or "Free"                                                      |
| Price          | character | Price of the app                                                             |
| Content Rating | character | The age group the app is targeted at - "Children" / "Mature 21+" / "Adult"   |
| Genres         | character | Possibly multiple genres the app belongs to                                  |
| Last Updated   | character | The date the app was last updated                                            |
| Current Ver    | character | The current version of the app                                               |
| Android Ver    | character | The Android version needed for this app                                      |

**data_reviews.csv**

| variable               | class        | description                                           |
|:-----------------------|:-------------|:------------------------------------------------------|
| App                    | character    | The application name                                  |
| Translated_Review      | character    | User review (translated to English)                   |
| Sentiment              | character    | The sentiment of the user - Positive/Negative/Neutral |
| Sentiment_Polarity     | character    | The sentiment polarity score                          |
| Sentiment_Subjectivity | character    | The sentiment subjectivity score                      |
