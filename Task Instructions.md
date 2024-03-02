## Basic tasks

There are in total seven basic tasks that every group must implement in order to have an accepted assignment.

The basic task 1 is a warming up task and it deals with some video game sales data. The task asks you to do some basic aggregation operations with Spark data frames.

The other basic tasks (basic tasks 2-7) are all related and deal with data from [https://moneypuck.com/data.htm](https://moneypuck.com/data.htm) that contains information about every shot in all National Hockey League ([NHL](https://en.wikipedia.org/wiki/National_Hockey_League), [ice hockey](https://en.wikipedia.org/wiki/Ice_hockey)) matches starting from season 2011-12 and ending with the last completed season, 2022-23. The tasks ask you to calculate the results of the matches based on the given data as well as do some further calculations. Knowledge about ice hockey or NHL is not required, and the task instructions should be sufficient in order to gain enough context for the tasks.

## Additional tasks

There are in total of three additional tasks that can be done to gain some course points.

The first additional task asks you to do all the basic tasks in an optimized way. It is possible that you can some points from this without directly trying by just implementing the basic tasks in an efficient manner.

The other two additional tasks are separate tasks and do not relate to any other basic or additional tasks. One of them asks you to load in unstructured text data and do some calculations based on the words found from the data. The other asks you to utilize the K-Means algorithm to partition the given building data.

It is possible to gain partial points from the additional tasks. I.e., if you have not completed the task fully but have implemented some part of the task, you might gain some appropriate portion of the points from the task.

## Basic Task 1 - Sales data

The CSV file `assignment/sales/video_game_sales.csv` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains video game sales data (from [https://www.kaggle.com/datasets/ashaheedq/video-games-sales-2019/data](https://www.kaggle.com/datasets/ashaheedq/video-games-sales-2019/data)). The direct address for the dataset is: `abfss://shared@tunics320f2023gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv`

Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset.

Only data for sales in the first ten years of the 21st century should be considered in this task, i.e. years 2000-2009.

Using the data, find answers to the following:

- Which publisher had the highest total sales in video games in European Union in years 2000-2009?
- What were the total yearly sales, in European Union and globally, for this publisher in year 2000-2009

## Basic Task 2 - Shot data from NHL matches

A parquet file in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/nhl_shots.parquet` from [https://moneypuck.com/data.htm](https://moneypuck.com/data.htm) contains information about every shot in all National Hockey League ([NHL](https://en.wikipedia.org/wiki/National_Hockey_League), [ice hockey](https://en.wikipedia.org/wiki/Ice_hockey)) matches starting from season 2011-12 and ending with the last completed season, 2022-23.

In this task you should load the data with all of the rows into a data frame. This data frame object will then be used in the following basic tasks 3-7.

### Background information

Each NHL season is divided into regular season and playoff season. In the regular season the teams play up to 82 games with the best teams continuing to the playoff season. During the playoff season the remaining teams are paired and each pair play best-of-seven series of games to determine which team will advance to the next phase.

In ice hockey each game has a home team and an away team. The regular length of a game is three 20 minute periods, i.e. 60 minutes or 3600 seconds. The team that scores more goals in the regulation time is the winner of the game.

If the scoreline is even after this regulation time:

- In playoff games, the game will be continued until one of the teams score a goal with the scoring team being the winner.
- In regular season games, there is an extra time that can last a maximum of 5 minutes (300 seconds). If one of the teams score, the game ends with the scoring team being the winner. If there is no goals in the extra time, there would be a shootout competition to determine the winner. These shootout competitions are not considered in this assignment, and the shots from those are not included in the raw data.

**Columns in the data**

Each row in the given data represents one shot in a game.

The column description from the source website. Not all of these will be needed in this assignment.

| column name | column type | description |
| ----------- | ----------- | ----------- |
| shotID      | integer | Unique id for each shot |
| homeTeamCode | string | The home team in the game. For example: TOR, MTL, NYR, etc. |
| awayTeamCode | string | The away team in the game |
| season | integer | Season the shot took place in. Example: 2009 for the 2009-2010 season |
| isPlayOffGame | integer | Set to 1 if a playoff game, otherwise 0 |
| game_id | integer | The NHL Game_id of the game the shot took place in |
| time | integer | Seconds into the game of the shot |
| period | integer | Period of the game |
| team | string | The team taking the shot. HOME or AWAY |
| location | string | The zone the shot took place in. HOMEZONE, AWAYZONE, or Neu. Zone |
| event | string | Whether the shot was a shot on goal (SHOT), goal, (GOAL), or missed the net (MISS) |
| homeTeamGoals | integer | Home team goals before the shot took place |
| awayTeamGoals | integer | Away team goals before the shot took place |
| homeTeamWon | integer | Set to 1 if the home team won the game. Otherwise 0. |
| shotType | string | Type of the shot. (Slap, Wrist, etc) |

## Basic Task 3 - Game data frame

Create a match data frame for all the game included in the shots data frame created in basic task 2.

The output should contain one row for each game.

The following columns should be included in the final data frame for this task:

| column name    | column type | description |
| -------------- | ----------- | ----------- |
| season         | integer     | Season the game took place in. Example: 2009 for the 2009-2010 season |
| game_id        | integer     | The NHL Game_id of the game |
| homeTeamCode   | string      | The home team in the game. For example: TOR, MTL, NYR, etc. |
| awayTeamCode   | string      | The away team in the game |
| isPlayOffGame  | integer     | Set to 1 if a playoff game, otherwise 0 |
| homeTeamGoals  | integer     | Number of goals scored by the home team |
| awayTeamGoals  | integer     | Number of goals scored by the away team |
| lastGoalTime   | integer     | The time in seconds for the last goal in the game. 0 if there was no goals in the game. |

All games had at least some shots but there are some games that did not have any goals either in the regulation 60 minutes or in the extra time.

Note, that for a couple of games there might be some shots, including goal-scoring ones, that are missing from the original dataset. For example, there might be a game with a final scoreline of 3-4 but only 6 of the goal-scoring shots are included in the dataset. Your solution does not have to try to take these rare occasions of missing data into account. I.e., you can do all the tasks with the assumption that there are no missing or invalid data.

## Basic Task 4 - Game wins during playoff seasons

Create a data frame that uses the game data frame from the basic task 3 and contains aggregated number of wins and losses for each team and for each playoff season, i.e. for games which have been marked as playoff games. Only teams that have played in at least one playoff game in the considered season should be included in the final data frame.

The following columns should be included in the final data frame:

| column name    | column type | description |
| -------------- | ----------- | ----------- |
| season         | integer     | The season for the data. Example: 2009 for the 2009-2010 season |
| teamCode       | string      | The code for the team. For example: TOR, MTL, NYR, etc. |
| games          | integer     | Number of playoff games the team played in the given season |
| wins           | integer     | Number of wins in playoff games the team had in the given season |
| losses         | integer     | Number of losses in playoff games the team had in the given season |

Playoff games where a team scored more goals than their opponent are considered winning games. And playoff games where a team scored less goals than the opponent are considered losing games.

In real life there should not be any playoff games where the final score line was even but due to some missing shot data you might end up with a couple of playoff games that seems to have ended up in a draw. These possible "drawn" playoff games can be left out from the win/loss calculations.

## Basic Task 5 - Best playoff teams

Using the playoff data frame created in basic task 4 create a data frame containing the win-loss record for best playoff team, i.e. the team with the most wins, for each season. You can assume that there are no ties for the highest amount of wins in each season.

The following columns should be included in the final data frame:

| column name    | column type | description |
| -------------- | ----------- | ----------- |
| season         | integer     | The season for the data. Example: 2009 for the 2009-2010 season |
| teamCode       | string      | The team code for the best performing playoff team in the given season. For example: TOR, MTL, NYR, etc. |
| games          | integer     | Number of playoff games the best performing playoff team played in the given season |
| wins           | integer     | Number of wins in playoff games the best performing playoff team had in the given season |
| losses         | integer     | Number of losses in playoff games the best performing playoff team had in the given season |

Finally, fetch the details for the best playoff team in season 2022.

## Basic Task 6 - Regular season points

Create a data frame that uses the game data frame from the basic task 3 and contains aggregated data for each team and for each season for the regular season matches, i.e. the non-playoff matches.

The following columns should be included in the final data frame:

| column name    | column type | description |
| -------------- | ----------- | ----------- |
| season         | integer     | The season for the data. Example: 2009 for the 2009-2010 season |
| teamCode       | string      | The code for the team. For example: TOR, MTL, NYR, etc. |
| games          | integer     | Number of non-playoff games the team played in the given season |
| wins           | integer     | Number of wins in non-playoff games the team had in the given season |
| losses         | integer     | Number of losses in non-playoff games the team had in the given season |
| goalsScored    | integer     | Total number goals scored by the team in non-playoff games in the given season |
| goalsConceded  | integer     | Total number goals scored against the team in non-playoff games in the given season |
| points         | integer     | Total number of points gathered by the team in non-playoff games in the given season |

Points from each match are received as follows (in the context of this assignment, these do not exactly match the NHL rules):

| points | situation |
| ------ | --------- |
| 3      | team scored more goals than the opponent during the regular 60 minutes |
| 2      | the score line was even after 60 minutes but the team scored a winning goal during the extra time |
| 1      | the score line was even after 60 minutes but the opponent scored a winning goal during the extra time or there were no goals in the extra time |
| 0      | the opponent scored more goals than the team during the regular 60 minutes |

In the regular season the following table shows how wins and losses should be considered (in the context of this assignment):

| win | loss | situation |
| --- | ---- | --------- |
| Yes | No   | team gained at least 2 points from the match |
| No  | Yes  | team gain at most 1 point from the match |

## Basic Task 7 - The worst regular season teams

Using the regular season data frame created in the basic task 6, create a data frame containing the regular season records for the worst regular season team, i.e. the team with the least amount of points, for each season. You can assume that there are no ties for the lowest amount of points in each season.

Finally, fetch the details for the worst regular season team in season 2022.

## Additional Task 1 - Optimized solutions to the basic tasks (2 points)

Use the tools Spark offers effectively and avoid unnecessary operations in the code for the basic tasks.

A couple of things to consider (**NOT** even close to a complete list):

- Consider using explicit schemas when dealing with CSV data sources.
- Consider only including those columns from a data source that are actually needed.
- Filter unnecessary rows whenever possible to get smaller datasets.
- Avoid collect or similar expensive operations for large datasets.
- Consider using explicit caching if some data frame is used repeatedly.
- Avoid unnecessary shuffling (for example sorting) operations.

It is okay to have your own test code that would fall into category of "ineffective usage" or "unnecessary operations" while doing the assignment tasks. However, for the final Moodle submission you should comment out or delete such code (and test that you have not broken anything when doing the final modifications).

Note, that you should not do the basic tasks again for this additional task, but instead modify your basic task code with more efficient versions.

You can create a text cell below this one and describe what optimizations you have done. This might help the grader to better recognize how skilled your work with the basic tasks has been.

## Additional Task 2 - Unstructured data

You are given some text files with contents from a few thousand random articles both in English and Finnish from Wikipedia. Content from English articles are in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/wikipedia/en` and content from Finnish articles are at folder `assignment/wikipedia/fi`.

Some cleaning operations have already been done to the texts but the some further cleaning is still required.

The final goal of the task is to get the answers to following questions:

- What are the ten most common English words that appear in the English articles?
- What are the five most common 5-letter Finnish words that appear in the Finnish articles?
- What is the longest word that appears at least 150 times in the articles?
- What is the average English word length for the words appearing in the English articles?
- What is the average Finnish word length for the words appearing in the Finnish articles?

For a word to be included in the calculations, it should fulfill the following requirements:

- Capitalization is to be ignored. I.e., words "English", "ENGLISH", and "english" are all to be considered as the same word "english".
- An English word should only contain the 26 letters from the alphabet of Modern English. Only exception is that punctuation marks, i.e. hyphens `-`, are allowed in the middle of the words as long as there are no two consecutive punctuation marks without any letters between them.
- The only allowed 1-letter English words are `a` and `i`.
- A Finnish word should follow the same rules as English words, except that three additional letters, `å`, `ä`, and `ö`, are also allowed, and that no 1-letter words are allowed. Also, any word that contains "`wiki`" should not be considered as Finnish word.

Some hints:

- Using an RDD or a Dataset (in Scala) might make the data cleaning and word determination easier than using DataFrames.
- It can be assumed that in the source data each word in the same line is separated by at least one white space (` `).
- You are allowed to remove all non-allowed characters from the source data at the beginning of the cleaning process.
- It is advisable to first create a DataFrame/Dataset/RDD that contains the found words, their language, and the number of times those words appeared in the articles. This can then be used as the starting point when determining the answers to the given questions.

## Additional Task 3 - K-Means clustering

You are given a dataset containing the locations of building in Finland. The dataset is a subset from [https://www.avoindata.fi/data/en_GB/dataset/postcodes/resource/3c277957-9b25-403d-b160-b61fdb47002f](https://www.avoindata.fi/data/en_GB/dataset/postcodes/resource/3c277957-9b25-403d-b160-b61fdb47002f) limited to only postal codes with the first two numbers in the interval 30-44 ([postal codes in Finland](https://www.posti.fi/en/zip-code-search/postal-codes-in-finland)). The dataset is in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/buildings.parquet`.

[K-Means clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm is an unsupervised machine learning algorithm that can be used to partition the input data into k clusters. Your task is to use the Spark ML library and its K-Means clusterization algorithm to divide the buildings into clusters using the building coordinates `latitude_wgs84` and `longitude_wgs84` as the basis of the clusterization. You should implement the following procedure:

1. Start with all the buildings in the dataset.
2. Divide the buildings into seven clusters with K-Means algorithm using `k=7` and the longitude and latitude of the buildings.
3. Find the cluster to which the Sähkötalo building from the Hervanta campus is sorted into. The building id for Sähkötalo in the dataset is `102363858X`.
4. Choose all the buildings from the cluster with the Sähkötalo building.
5. Find the cluster center for the chosen cluster of buildings.
6. Calculate the largest distance from a building in the chosen cluster to the chosen cluster center. You are given a function `haversine` that you can use to calculate the distance between two points using the latitude and longitude of the points.
7. While the largest distance from a building in the considered cluster to the cluster center is larger than 3 kilometers run the K-Means algorithm again using the following substeps.
    - Run the K-Means algorithm to divide the remaining buildings into smaller clusters. The number of the new clusters should be one less than in the previous run of the algorithm (but should always be at least two). I.e., the sequence of `k` values starting from the second run should be 6, 5, 4, 3, 2, 2, ...
    - After using the algorithm again, choose the new smaller cluster of buildings so that it includes the Sähkötalo building.
    - Find the center of this cluster and calculate the largest distance from a building in this cluster to its center.

As the result of this process, you should get a cluster of buildings that includes the Sähkötalo building and in which all buildings are within 3 kilometers of the cluster center.

Using the final cluster, find the answers to the following questions:

- How many buildings in total are in the final cluster?
- How many Hervanta buildings are in this final cluster? (A building is considered to be in Hervanta if their postal code is `33720`)

Some hints:

- Once you have trained a KMeansModel, the coordinates for the cluster centers, and the cluster indexes for individual buildings can be accessed through the model object (`clusterCenters`, `summary.predictions`).
- The given haversine function for calculating distances can be used with data frames if you turn it into an user defined function.
