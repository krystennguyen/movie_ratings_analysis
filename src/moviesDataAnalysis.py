#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:34:28 2024

@author: krysten
"""

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os
from scipy import stats


df = pd.read_csv('../data/movieReplicationSet.csv', header=0)

# declare the number of movies (index of the last column with movie ratings is N_MOVIES - 1)
N_MOVIES = 400
movies_df = df.iloc[:, :N_MOVIES]  # df with only 400 movie rating columns

# Ratings data is is nominally and ordinally interpretable -> use Mann-Whitney U test if 2 groups - Kruskal-Wallis if 3+ groups
# if comparing distribution -> use KS test

# graph the distribution of the sample to see if median comparison is reasonable
total_n = movies_df.notna().sum().sum()
ratings_pp = movies_df.apply(lambda row: row.notna().sum(), axis=1)
plt.figure()
plt.hist(ratings_pp, bins=100)
plt.title(
    f'Distribution of ratings per participant \n Total number of ratings: {total_n}')
plt.xlabel('Number of ratings')
plt.ylabel('Frequency')
plt.show()
plt.savefig('../results/figures/ratings_per_participant.png')


############################
# 1) Are movies that are more popular (operationalized as having more ratings) rated higher than movies that are less popular? [Hint: You can do a median-split of popularity to determinehigh vs. low popularitymovies]

# get number of ratings from each movies
n_ratings = movies_df.apply(lambda col: col.notna().sum(), axis=0)
median_n_rating = n_ratings.median()

# get high and low popularity movies at the median cutoff
high_pop = movies_df.loc[:, n_ratings > median_n_rating]
low_pop = movies_df.loc[:, n_ratings <= median_n_rating]

# flatten into 1D and get rid of nan
high_pop_flat = pd.Series(high_pop.values.flatten())
low_pop_flat = pd.Series(low_pop.values.flatten())

uTest_popular = stats.mannwhitneyu(
    high_pop_flat, low_pop_flat, alternative='greater', nan_policy='omit')
print('Q1: Popular versus less popular movie ratings: ', uTest_popular)

# graph the distribution of ratings for high and low popularity movies
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.hist(high_pop_flat, label='High popularity', color='purple')
plt.title('Rating distribution of high popularity movies \n Median rating: ' +
          str(high_pop_flat.median()))
plt.subplot(2, 1, 2)
plt.hist(low_pop_flat, label='Low popularity', color='green')
plt.title('Rating distribution of low popularity movies \n Median rating: ' +
          str(low_pop_flat.median()))
plt.savefig('../results/figures/high_low_popularity_distribution.png')
plt.show()


############################
# 2) Are movies that are newer rated differently than movies that are older? [Hint: Do a median split of year of release to contrast movies in terms of whether they are old or new]
years_released = movies_df.columns.str.extract(r'\((\d{4})\)').astype(int)
years_released = pd.Series(years_released.values.flatten())
years_released.index = movies_df.columns
median_year = years_released.median()

new = movies_df.loc[:, years_released > median_year]
old = movies_df.loc[:, years_released <= median_year]

new_flat = pd.Series(new.values.flatten())
old_flat = pd.Series(old.values.flatten())

uTest_new = stats.mannwhitneyu(new_flat, old_flat, nan_policy='omit')
print('Q2: Older versus newer movie: ', uTest_new)
print('\n')


############################
# 3) Is enjoyment of ‘Shrek(2001)’ gendered, i.e. do male and female viewers rate it differently?
shrek_gender_df = df.filter(regex='Shrek \(2001\)|Gender identity')


shrek_female_ratings = shrek_gender_df[shrek_gender_df[
    'Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1]['Shrek (2001)']
shrek_male_ratings = shrek_gender_df[shrek_gender_df[
    'Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2]['Shrek (2001)']

uTest_shrek = stats.mannwhitneyu(shrek_female_ratings.values.flatten(
), shrek_male_ratings.values.flatten(), nan_policy='omit', alternative='two-sided')
print('Q3: Shrek ratings between male and female: ', uTest_shrek)
print('\n')


############################
# 4) What proportion of movies are rated differently by male and female viewers?
# run U test for each movie and count the number of significant results out of all movies

gender_split = {}
significance_count_gender = 0
for movie in movies_df:
    female_ratings = df[df['Gender identity (1 = female; 2 = male; 3 = self-described)']
                        == 1][movie]
    male_ratings = df[df['Gender identity (1 = female; 2 = male; 3 = self-described)']
                      == 2][movie]
    uTest_gender = stats.mannwhitneyu(female_ratings.values.flatten(
    ), male_ratings.values.flatten(), nan_policy='omit', alternative='two-sided')
    p_value = uTest_gender.pvalue
    gender_split[movie] = p_value

    if p_value < 0.005:
        significance_count_gender += 1

top5_gender_diff = sorted(gender_split, key=gender_split.get, reverse=True)[:5]

print('Q4: Number of movies rated differently between male and female viewers (alpha level: 0.005): ', significance_count_gender)
print('\tProportion: ', significance_count_gender/N_MOVIES * 100, '%')
print('\tTop 5 movies with the most significant difference in ratings between gender: \n', top5_gender_diff)
print('\n')

# save the results to a csv file
gender_split_df = pd.DataFrame.from_dict(gender_split, orient='index', columns=['p-value'])
gender_split_df.to_csv('../results/significance_tests/gender_difference.csv')

############################
# 5) Do people who are only children enjoy ‘The Lion King(1994)’ more than people with siblings?
lion_king_df = df.filter(regex='The Lion King \(1994\)|only child')
only_child_lk = lion_king_df[lion_king_df['Are you an only child? (1: Yes; 0: No; -1: Did not respond)']
                             == 1]['The Lion King (1994)']
sibling_lk = lion_king_df[lion_king_df['Are you an only child? (1: Yes; 0: No; -1: Did not respond)']
                          == 0]['The Lion King (1994)']

uTest_lionking = stats.mannwhitneyu(only_child_lk.values.flatten(
), sibling_lk.values.flatten(), nan_policy='omit', alternative='two-sided')
print('Q5: Lion King ratings between only child and people with siblings: ', uTest_lionking)
print('\n')


############################
# 6) What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings vs. those without?
# U test for each film
only_child_effect = {}
significance_count_onlychild = 0

print('Q6: Movie with only child effect (alpha level: 0.005): ')
i =1
for movie in movies_df:
    only_child_ratings = df[df['Are you an only child? (1: Yes; 0: No; -1: Did not respond)']
                            == 1][movie]
    sibling_ratings = df[df['Are you an only child? (1: Yes; 0: No; -1: Did not respond)']
                         == 0][movie]
    uTest_gender = stats.mannwhitneyu(only_child_ratings.values.flatten(
    ), sibling_ratings.values.flatten(), nan_policy='omit', alternative='two-sided')
    p_value = uTest_gender.pvalue
    only_child_effect[movie] = p_value

    if p_value < 0.005:
        significance_count_onlychild += 1
        print(i,'. ', movie)
        i+=1

print('\tProportion: ', round(significance_count_onlychild/N_MOVIES * 100, 2), '%')
print('\n')

# save the results to a csv file
only_child_effect_df = pd.DataFrame.from_dict(only_child_effect, orient='index', columns=['p-value'])
only_child_effect_df.to_csv('../results/significance_tests/only_child_effect.csv')


############################
# 7) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who preferto watch them alone?
wolf_df = df.filter(
    regex='The Wolf of Wall Street \(2013\)|Movies are best enjoyed alone')
alone_wolf = wolf_df[wolf_df['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']
                     == 1]['The Wolf of Wall Street (2013)']
social_wolf = wolf_df[wolf_df['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']
                      == 0]['The Wolf of Wall Street (2013)']

uTest_wolf = stats.mannwhitneyu(alone_wolf.values.flatten(
), social_wolf.values.flatten(), nan_policy='omit', alternative='two-sided')
print('Q7: Wolf of Wall Street ratings between people who watch movies alone and socially: ', uTest_wolf)
print('\n')


############################
# 8) What proportion of movies exhibit such a “social watching” effect?
social_watching_effect = {}
significance_count_social = 0
i=0
print('Q8: Movie with social watching effect (alpha level: 0.005): ')
for movie in movies_df:
    alone_ratings = df[df['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']
                       == 1][movie]
    social_ratings = df[df['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)']
                        == 0][movie]
    uTest_social = stats.mannwhitneyu(alone_ratings.values.flatten(
    ), social_ratings.values.flatten(), nan_policy='omit', alternative='two-sided')
    p_value = uTest_social.pvalue
    social_watching_effect[movie] = p_value
    if p_value < 0.005:
        significance_count_social += 1
        print(i,'. ', movie)
        i+=1
print('\tProportion:', significance_count_social/N_MOVIES * 100, '%')
print('\n')

# save the results to a csv file
social_watching_effect_df = pd.DataFrame.from_dict(social_watching_effect, orient='index', columns=['p-value'])
social_watching_effect_df.to_csv('../results/significance_tests/social_watching_effect.csv')


############################
# 9) Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?
# KS test for comparing distribution
home_alone = movies_df['Home Alone (1990)'].dropna()
finding_nemo = movies_df['Finding Nemo (2003)'].dropna()
ks_test = stats.ks_2samp(home_alone, finding_nemo)
print('Q9: Home Alone and Finding Nemo ratings distribution: ', ks_test)


# graph the rating distribution of the two movies

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.hist(home_alone, label='Home Alone', color='purple')
plt.title('Rating distribution of Home Alone (1990) \n Number of ratings: ' +
          str(len(home_alone)))
plt.subplot(2, 1, 2)
plt.hist(finding_nemo, label='Finding Nemo', color='green')
plt.title('Rating distribution of Finding Nemo (2003) \n Number of ratings: ' +
          str(len(finding_nemo)))
plt.savefig('../results/figures/home_alone_finding_nemo_distribution.png')
plt.show()


############################
# 10) There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) inthis dataset. How many of these are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks featured in this question to identify the movies that are part of each franchise]
# Kruskal Wallis for 3+ groups
franchise_ratings = {}
franchise_inconsistent_count = 0
franchises = ['Star Wars', 'Harry Potter', 'The Matrix', 'Indiana Jones',
              'Jurassic Park', 'Pirates of the Caribbean', 'Toy Story', 'Batman']
for franchise in franchises:
    franchise_df = movies_df.filter(regex=franchise)
    kruskal_test = stats.kruskal(
        *[franchise_df[col] for col in franchise_df.columns], nan_policy='omit')
    franchise_ratings[franchise] = kruskal_test.pvalue
    if kruskal_test.pvalue < 0.005:
        franchise_inconsistent_count += 1


print('Q10: Number of inconsistent quality franchises (alpha level: 0.005): ',
      franchise_inconsistent_count, 'out out of', len(franchises), 'franchises')
print('\tHarry Potter is the only franchise with consistent rating, at p-value of',
      franchise_ratings['Harry Potter'])
print('\n')

# save the results to a csv file
franchise_ratings_df = pd.DataFrame.from_dict(franchise_ratings, orient='index', columns=['p-value'])
franchise_ratings_df.to_csv('../results/significance_tests/franchise_ratings_consistencies.csv')



############################
# Extra Credit: Tell us something interesting and true (supported by a significance test of some kind) about the movies in this dataset that is not already covered by the questions above [for 5% of the grade score].
# What proportion of movies are rated highly by viewers self-identified as sophisticated in art, music, or literature?

sophisticated_df = df.filter(regex='sophisticated')

median_soph = sophisticated_df.median()[0]

movies_soph_df = pd.concat([movies_df, sophisticated_df], axis=1)

print('Extra Credit: Movies rated highly by self-identified artistically sophisticated and non-sophisticated viewers (alpha level: 0.005): ')

significance_count_soph = 0
sophisticated_split = {}
i=1

for movie in movies_df:
    sophisticated = movies_soph_df[movies_soph_df[sophisticated_df.columns[0]]
                                   > median_soph][movie]
    not_sophisticated = movies_soph_df[movies_soph_df[sophisticated_df.columns[0]]
                                       <= median_soph][movie]
    uTest_soph = stats.mannwhitneyu(sophisticated.values.flatten(
    ), not_sophisticated.values.flatten(), nan_policy='omit', alternative='greater')
    p_value = uTest_soph.pvalue
    sophisticated_split[movie] = p_value
    if p_value < 0.005:
        significance_count_soph += 1
        print(i,'. ', movie)
        i+=1

print('\tProportion', significance_count_soph/N_MOVIES * 100, '%')

# save the results to a csv file
sophisticated_split_df = pd.DataFrame.from_dict(sophisticated_split, orient='index', columns=['p-value'])
sophisticated_split_df.to_csv('../results/significance_tests/sophisticated_favoring_effect.csv')


