# As the main metric for evaluating my system I decided to use NDCG score.
# I chose it because of non-binary notions of relevance, in our case ratings.
import pickle
import pandas as pd

pd.set_option('display.max_colwidth', None)
from sklearn import preprocessing as pp
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import ndcg_score


def make_dfs():
    columns_name = ['user_id', 'item_id', 'rating', 'timestamp']
    train_df = pd.read_csv("../data/raw/ml-100k/ua.base", sep="\t", names=columns_name)
    test_df = pd.read_csv("../data/raw/ml-100k/ua.test", sep="\t", names=columns_name)

    return train_df, test_df


def preproc(train_df, test_df):
    film_columns = ["item_id", "movie title", "release date", "video release date",
                    "IMDb URL", "unknown", "Action", "Adventure", "Animation",
                    "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                    "Thriller", "War", "Western"]

    films_df = pd.read_csv("../data/raw/ml-100k/u.item", sep="|", names=film_columns, encoding='latin-1')
    films_df.drop(["movie title", "release date", "IMDb URL", "unknown", "video release date"], axis=1, inplace=True)

    train_df = pd.merge(train_df, films_df, how='left', left_on='item_id', right_on='item_id')
    test_df = pd.merge(test_df, films_df, how='left', left_on='item_id', right_on='item_id')

    user_columns = ["user_id", "age", "sex", "occupation", "zip_code"]
    user_df = pd.read_csv("../data/raw/ml-100k/u.user", sep="|", names=user_columns, encoding='latin-1')
    user_df["sex"] = pp.LabelEncoder().fit_transform(user_df["sex"])
    occup_df = pd.read_csv("../data/raw/ml-100k/u.occupation", sep="\t", names=["jobs"])
    le = pp.LabelEncoder()
    le.fit(occup_df["jobs"])
    user_df["occupation"] = le.transform(user_df["occupation"])
    user_df.drop(["zip_code"], axis=1, inplace=True)

    train_df = pd.merge(train_df, user_df, how='left', left_on='user_id', right_on='user_id')
    test_df = pd.merge(test_df, user_df, how='left', left_on='user_id', right_on='user_id')

    train_df.drop(["item_id", "user_id", "timestamp"], axis=1, inplace=True)
    train_y = train_df["rating"].values
    train_x = train_df.drop('rating', axis=1).values

    test_df.drop(["item_id", "user_id", "timestamp"], axis=1, inplace=True)
    test_y = test_df["rating"].values
    test_x = test_df.drop('rating', axis=1).values

    return train_x, train_y, test_x, test_y


train_df, test_df = make_dfs()
train_x, train_y, test_x, test_y = preproc(train_df, test_df)  # preprocess and divide data

test_user_lines = {}  # Getting useful data for recommendation
test_items = []
for i, data in test_df.iterrows():
    test_items.append(data["item_id"])
    if data["user_id"] not in test_user_lines.keys():
        test_user_lines[data["user_id"]] = [i]
    else:
        test_user_lines[data["user_id"]].append(i)

# Loading model
clf = pickle.load(open("../models/finalized_random_forest", 'rb'))

preds = []  # Predictions
for i in range(len(test_y)):
    pred = clf.predict(test_x[i, :].reshape(1, -1))
    preds.append(pred[0])


def find_ndcg(user_id):
    predictions = [[clf.predict(test_x[j, :].reshape(1, -1))[0] for j in test_user_lines[user_id]]]
    real_rating = [[test_y[j] for j in test_user_lines[user_id]]]

    return ndcg_score(real_rating, predictions)


def evaluate():  # Compute mean NDCG score
    ndcg = 0
    total = 0
    for i in test_user_lines.keys():
        total += 1
        ndcg += find_ndcg(i)
    return ndcg / total


print("Model evaluation:")
print("Mean absolute error: ", mean_absolute_error(test_y, preds))
print("Mean ndcg score: ", evaluate())  # As you see, result is not bad


def recommend_10(user_id):  # Since for testing I use ua.test, where each user got
    # exactly 10 ratings, I will recommend user 10 movies based on my predicted ratings
    # of movies
    predictions = [clf.predict(test_x[j, :].reshape(1, -1))[0] for j in test_user_lines[user_id]]
    real_rating = [test_y[j] for j in test_user_lines[user_id]]  # Ground truth ratings

    recommendations = [[test_items[i]] for i in test_user_lines[user_id]]
    for i in range(len(predictions)):
        recommendations[i].append(predictions[i])
    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort movies according to predicted ratings

    ideal_recommendations = [[test_items[i]] for i in test_user_lines[user_id]]
    for i in range(len(real_rating)):
        ideal_recommendations[i].append(real_rating[i])
    ideal_recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort movies according to real ratings

    print("My recommendations: ", [i[0] for i in recommendations])
    print("Ideal recommendations: ", [i[0] for i in ideal_recommendations])


print("\nExample of recommendation:")
print("Recommending 10 movies for User #8")
recommend_10(8)  # Recommend 10 movies for user #8

# As you see, my recommendations are not far from ideal ones, based on the test data
