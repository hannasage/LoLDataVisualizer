import numpy as np
import pandas as pd
import Keys as Key
import Evaluation as Ev

player_df = pd.read_csv("C:/Users/Kevin/PycharmProjects/leagueoflegends/PlayerData.csv")
team_df = pd.read_csv("C:/Users/Kevin/PycharmProjects/leagueoflegends/TeamData.csv")

# Clears incorrect index column created by Pandas
Ev.clear_false_index(player_df, team_df)

"""
DO ALL DATA ANALYSIS BELOW
"""

# Example of make_sub_set()
test_df = Ev.make_sub_set([Key.gold_at_10, Key.team], team_df)

# Example of simple_feature_scale()
test_df[Key.gold_at_10 + '-normalized'] = Ev.simple_feature_scale(Key.gold_at_10, test_df)

# Example of create_binned_column() [function also utilizes create_bins()]
test_df[Key.gold_at_10 + '-binned'] = Ev.create_binned_column(Key.gold_at_10,
                                                              ['Low', 'Medium', 'High'],
                                                              test_df)
print(test_df)
