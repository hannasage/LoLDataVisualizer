import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Keys as Key
import Evaluation as Ev
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

player_df = pd.read_csv("C:/Users/Kevin/PycharmProjects/leagueoflegends/PlayerData.csv")
team_df = pd.read_csv("C:/Users/Kevin/PycharmProjects/leagueoflegends/TeamData.csv")

# Clears incorrect index column created by Pandas
Ev.clear_false_index(player_df, team_df)

"""
DO ALL DATA ANALYSIS BELOW
"""

# Example of make_sub_set()
# test_df = Ev.make_sub_set([Key.gold_at_10, Key.team], team_df)

# Example of simple_feature_scale()
# test_df[Key.gold_at_10 + '-normalized'] = Ev.simple_feature_scale(Key.gold_at_10, test_df)

# Example of create_binned_column() [function also utilizes create_bins()]
# test_df[Key.gold_at_10 + '-binned'] = Ev.create_binned_column(Key.gold_at_10,
#                                                               ['Low', 'Medium', 'High'],
#                                                               test_df)


# PLOTTING MLR
subset = team_df.dropna()
lm = LinearRegression()

x1 = subset[[Key.first_baron_time, Key.team_baron_kills, Key.dmg_to_champs_per_min,
             Key.wards_per_min, Key.cs_per_min]]
y = subset[Key.game_length]

lm.fit(x1, y)

score = lm.score(x1, y)
Yhat = lm.predict(x1)
mse = mean_squared_error(subset[Key.result], Yhat)
print("The R^2 is: {}".format(score))
print("The first {} values of Yhat are: {}".format(5, Yhat[0:5]))
print("The MSE of Result and Predicted Values is: {}".format(mse))
print("Coef_: {}, Intercept: {}".format(lm.coef_, lm.intercept_))

width = 6
height = 5
plt.figure(figsize=(width, height))

# LINEAR REGRESSION
ax1 = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)
ax1.set_yticklabels([])

plt.title('Actual vs Fitted Values: Game Length')
plt.xlabel('Game Length')


# HEATMAP
# data = pd.concat([x1, y])
# ax = sns.heatmap(data, annot=True, annot_kws={"size": 7}, square=True, vmin=0, vmax=1)
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)

plt.show()

