# Databricks notebook source
# MAGIC %md
# MAGIC # Data-Intensive Programming - Assignment
# MAGIC
# MAGIC This is the **Python** version of the assignment. Switch to the Scala version, if you want to do the assignment in Scala.
# MAGIC
# MAGIC In all tasks, add your solutions to the cells following the task instructions. You are free to add new cells if you want.
# MAGIC
# MAGIC Don't forget to **submit your solutions to Moodle** once your group is finished with the assignment.
# MAGIC
# MAGIC ## Basic tasks (compulsory)
# MAGIC
# MAGIC There are in total seven basic tasks that every group must implement in order to have an accepted assignment.
# MAGIC
# MAGIC The basic task 1 is a warming up task and it deals with some video game sales data. The task asks you to do some basic aggregation operations with Spark data frames.
# MAGIC
# MAGIC The other basic tasks (basic tasks 2-7) are all related and deal with data from [https://moneypuck.com/data.htm](https://moneypuck.com/data.htm) that contains information about every shot in all National Hockey League ([NHL](https://en.wikipedia.org/wiki/National_Hockey_League), [ice hockey](https://en.wikipedia.org/wiki/Ice_hockey)) matches starting from season 2011-12 and ending with the last completed season, 2022-23. The tasks ask you to calculate the results of the matches based on the given data as well as do some further calculations. Knowledge about ice hockey or NHL is not required, and the task instructions should be sufficient in order to gain enough context for the tasks.
# MAGIC
# MAGIC ## Additional tasks (optional, can provide course points)
# MAGIC
# MAGIC There are in total of three additional tasks that can be done to gain some course points.
# MAGIC
# MAGIC The first additional task asks you to do all the basic tasks in an optimized way. It is possible that you can some points from this without directly trying by just implementing the basic tasks in an efficient manner.
# MAGIC
# MAGIC The other two additional tasks are separate tasks and do not relate to any other basic or additional tasks. One of them asks you to load in unstructured text data and do some calculations based on the words found from the data. The other asks you to utilize the K-Means algorithm to partition the given building data.
# MAGIC
# MAGIC It is possible to gain partial points from the additional tasks. I.e., if you have not completed the task fully but have implemented some part of the task, you might gain some appropriate portion of the points from the task.
# MAGIC

# COMMAND ----------

# import statements for the entire notebook
# add anything that is required here

from dataclasses import dataclass
import math
from typing import List, Optional

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.rdd import RDD
from pyspark.sql import Column, DataFrame, Row
from pyspark.sql.udf import UserDefinedFunction
from pyspark.sql.window import Window, WindowSpec
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from pyspark.storagelevel import StorageLevel


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 1 - Sales data
# MAGIC
# MAGIC The CSV file `assignment/sales/video_game_sales.csv` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains video game sales data (from [https://www.kaggle.com/datasets/ashaheedq/video-games-sales-2019/data](https://www.kaggle.com/datasets/ashaheedq/video-games-sales-2019/data)). The direct address for the dataset is: `abfss://shared@tunics320f2023gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv`
# MAGIC
# MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset.
# MAGIC
# MAGIC Only data for sales in the first ten years of the 21st century should be considered in this task, i.e. years 2000-2009.
# MAGIC
# MAGIC Using the data, find answers to the following:
# MAGIC
# MAGIC - Which publisher had the highest total sales in video games in European Union in years 2000-2009?
# MAGIC - What were the total yearly sales, in European Union and globally, for this publisher in year 2000-2009
# MAGIC

# COMMAND ----------

salesSchema: StructType = StructType(
    [
        StructField("Rank", IntegerType(), False),
        StructField("Name", StringType(), False),
        StructField("Platform", StringType(), False),
        StructField("Year", IntegerType(), True),
        StructField("Genre", StringType(), False),
        StructField("Publisher", StringType(), False),
        StructField("NA_Sales", DoubleType(), False),
        StructField("EU_Sales", DoubleType(), False),
        StructField("JP_Sales", DoubleType(), False),
        StructField("Other_Sales", DoubleType(), False),
        StructField("Global_Sales", DoubleType(), False)
    ]
)

salesDF: DataFrame = (
    spark
        .read
        .option("header", True)
        .option("sep", ",")
        .schema(salesSchema)
        .csv("abfss://shared@tunics320f2023gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv")
        .filter((F.col("Year") >= 2000) & (F.col("Year") <= 2009))
        .select("Publisher", "Year", "EU_Sales", "Global_Sales")
        # .persist(StorageLevel.MEMORY_ONLY)
)

bestEUPublisher: str = (
    salesDF
        .select("Publisher", "EU_Sales")
        .groupBy("Publisher")
        .agg(F.sum("EU_Sales").alias("EU_Total"))
        .select(F.max_by(F.col("Publisher"), F.col("EU_Total")))
        .first()
        [0]
)

bestEUPublisherSales: DataFrame = (
    salesDF
        .filter(F.col("Publisher") == bestEUPublisher)
        .groupBy("Year")
        .agg(
            F.round(F.sum("EU_Sales"), 2).alias("EU_Total"),
            F.round(F.sum("Global_Sales"), 2).alias("Global_Total")
        )
        .orderBy(F.col("Year").asc())
)

print(f"The publisher with the highest total video game sales in European Union is: '{bestEUPublisher}'")
print("Sales data for the publisher:")
bestEUPublisherSales.show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 2 - Shot data from NHL matches
# MAGIC
# MAGIC A parquet file in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/nhl_shots.parquet` from [https://moneypuck.com/data.htm](https://moneypuck.com/data.htm) contains information about every shot in all National Hockey League ([NHL](https://en.wikipedia.org/wiki/National_Hockey_League), [ice hockey](https://en.wikipedia.org/wiki/Ice_hockey)) matches starting from season 2011-12 and ending with the last completed season, 2022-23.
# MAGIC
# MAGIC In this task you should load the data with all of the rows into a data frame. This data frame object will then be used in the following basic tasks 3-7.
# MAGIC
# MAGIC ### Background information
# MAGIC
# MAGIC Each NHL season is divided into regular season and playoff season. In the regular season the teams play up to 82 games with the best teams continuing to the playoff season. During the playoff season the remaining teams are paired and each pair play best-of-seven series of games to determine which team will advance to the next phase.
# MAGIC
# MAGIC In ice hockey each game has a home team and an away team. The regular length of a game is three 20 minute periods, i.e. 60 minutes or 3600 seconds. The team that scores more goals in the regulation time is the winner of the game.
# MAGIC
# MAGIC If the scoreline is even after this regulation time:
# MAGIC
# MAGIC - In playoff games, the game will be continued until one of the teams score a goal with the scoring team being the winner.
# MAGIC - In regular season games, there is an extra time that can last a maximum of 5 minutes (300 seconds). If one of the teams score, the game ends with the scoring team being the winner. If there is no goals in the extra time, there would be a shootout competition to determine the winner. These shootout competitions are not considered in this assignment, and the shots from those are not included in the raw data.
# MAGIC
# MAGIC **Columns in the data**
# MAGIC
# MAGIC Each row in the given data represents one shot in a game.
# MAGIC
# MAGIC The column description from the source website. Not all of these will be needed in this assignment.
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | shotID      | integer | Unique id for each shot |
# MAGIC | homeTeamCode | string | The home team in the game. For example: TOR, MTL, NYR, etc. |
# MAGIC | awayTeamCode | string | The away team in the game |
# MAGIC | season | integer | Season the shot took place in. Example: 2009 for the 2009-2010 season |
# MAGIC | isPlayOffGame | integer | Set to 1 if a playoff game, otherwise 0 |
# MAGIC | game_id | integer | The NHL Game_id of the game the shot took place in |
# MAGIC | time | integer | Seconds into the game of the shot |
# MAGIC | period | integer | Period of the game |
# MAGIC | team | string | The team taking the shot. HOME or AWAY |
# MAGIC | location | string | The zone the shot took place in. HOMEZONE, AWAYZONE, or Neu. Zone |
# MAGIC | event | string | Whether the shot was a shot on goal (SHOT), goal, (GOAL), or missed the net (MISS) |
# MAGIC | homeTeamGoals | integer | Home team goals before the shot took place |
# MAGIC | awayTeamGoals | integer | Away team goals before the shot took place |
# MAGIC | homeTeamWon | integer | Set to 1 if the home team won the game. Otherwise 0. |
# MAGIC | shotType | string | Type of the shot. (Slap, Wrist, etc) |
# MAGIC

# COMMAND ----------

shotsDF: DataFrame = (
    spark
        .read
        .parquet("abfss://shared@tunics320f2023gen2.dfs.core.windows.net/assignment/nhl_shots.parquet")
        .select(
            "season", "game_id", "homeTeamCode", "awayTeamCode", "isPlayOffGame",
            "team", "event", "time", "homeTeamGoals", "awayTeamGoals"
        )
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 3 - Game data frame
# MAGIC
# MAGIC Create a match data frame for all the game included in the shots data frame created in basic task 2.
# MAGIC
# MAGIC The output should contain one row for each game.
# MAGIC
# MAGIC The following columns should be included in the final data frame for this task:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | season         | integer     | Season the game took place in. Example: 2009 for the 2009-2010 season |
# MAGIC | game_id        | integer     | The NHL Game_id of the game |
# MAGIC | homeTeamCode   | string      | The home team in the game. For example: TOR, MTL, NYR, etc. |
# MAGIC | awayTeamCode   | string      | The away team in the game |
# MAGIC | isPlayOffGame  | integer     | Set to 1 if a playoff game, otherwise 0 |
# MAGIC | homeTeamGoals  | integer     | Number of goals scored by the home team |
# MAGIC | awayTeamGoals  | integer     | Number of goals scored by the away team |
# MAGIC | lastGoalTime   | integer     | The time in seconds for the last goal in the game. 0 if there was no goals in the game. |
# MAGIC
# MAGIC All games had at least some shots but there are some games that did not have any goals either in the regulation 60 minutes or in the extra time.
# MAGIC
# MAGIC Note, that for a couple of games there might be some shots, including goal-scoring ones, that are missing from the original dataset. For example, there might be a game with a final scoreline of 3-4 but only 6 of the goal-scoring shots are included in the dataset. Your solution does not have to try to take these rare occasions of missing data into account. I.e., you can do all the tasks with the assumption that there are no missing or invalid data.
# MAGIC

# COMMAND ----------

# helper variables for the task
gameWindow: WindowSpec = Window.partitionBy("season", "game_id")

goalEvent: Column = F.col("event") == "GOAL"
homeTeam: Column = F.col("team") == "HOME"
awayTeam: Column = F.col("team") == "AWAY"


# COMMAND ----------

# First, modify the goal columns to have the goal situation after the event and zero all times for non-goal events.
# Then, group by season and game_id and get the results by taking the maximum values.
gamesDF: DataFrame = (
    shotsDF
        .select(
            F.col("season"),
            F.col("game_id"),
            F.col("homeTeamCode"),
            F.col("awayTeamCode"),
            F.col("isPlayOffGame"),
            F.when(goalEvent & homeTeam, F.col("homeTeamGoals") + 1).otherwise(F.col("homeTeamGoals")).alias("homeTeamGoals"),
            F.when(goalEvent & awayTeam, F.col("awayTeamGoals") + 1).otherwise(F.col("awayTeamGoals")).alias("awayTeamGoals"),
            F.when(goalEvent, F.col("time")).otherwise(0).alias("time")
        )
        .groupBy("season", "game_id")
        .agg(
            F.first(F.col("homeTeamCode")).alias("homeTeamCode"),
            F.first(F.col("awayTeamCode")).alias("awayTeamCode"),
            F.first(F.col("isPlayOffGame")).alias("isPlayOffGame"),
            F.max("homeTeamGoals").alias("homeGoals"),
            F.max("awayTeamGoals").alias("awayGoals"),
            F.max("time").alias("lastGoalTime")
        )
)


# COMMAND ----------

# Alternative solution, in which the result is calculated by counting the goal-events.
gamesDF2: DataFrame = (
    shotsDF
        .select(
            F.col("season"),
            F.col("game_id"),
            F.col("homeTeamCode"),
            F.col("awayTeamCode"),
            F.col("isPlayOffGame"),
            F.when(goalEvent & homeTeam, 1).otherwise(0).alias("homeGoal"),
            F.when(goalEvent & awayTeam, 1).otherwise(0).alias("awayGoal"),
            F.when(goalEvent, F.col("time")).otherwise(0).alias("time")
        )
        .groupBy("season", "game_id")
        .agg(
            F.first(F.col("homeTeamCode")).alias("homeTeamCode"),
            F.first(F.col("awayTeamCode")).alias("awayTeamCode"),
            F.first(F.col("isPlayOffGame")).alias("isPlayOffGame"),
            F.sum(F.col("homeGoal")).alias("homeGoals"),
            F.sum(F.col("awayGoal")).alias("awayGoals"),
            F.max("time").alias("lastGoalTime")
        )
)


# COMMAND ----------

# Second alternative solution, in which the games with goals are handled separately from the games without goals.
# With the given data, this solution might be slightly less efficient than the others.

gamesWithGoalsDF: DataFrame = (
    shotsDF
        .filter(goalEvent)
        .select(
            F.col("season"),
            F.col("game_id"),
            F.col("homeTeamCode"),
            F.col("awayTeamCode"),
            F.col("isPlayOffGame"),
            F.when(homeTeam, 1).otherwise(0).alias("homeGoal"),
            F.when(awayTeam, 1).otherwise(0).alias("awayGoal"),
            F.col("time"),
        )
        .groupBy("season", "game_id")
        .agg(
            F.first(F.col("homeTeamCode")).alias("homeTeamCode"),
            F.first(F.col("awayTeamCode")).alias("awayTeamCode"),
            F.first(F.col("isPlayOffGame")).alias("isPlayOffGame"),
            F.sum(F.col("homeGoal")).alias("homeGoals"),
            F.sum(F.col("awayGoal")).alias("awayGoals"),
            F.max("time").alias("lastGoalTime")
        )
)

gamesWithoutGoalsDF: DataFrame = (
    shotsDF
        .withColumn("goals", F.count(F.when(goalEvent, 1)).over(gameWindow))
        .filter(F.col("goals") == 0)
        .select("season", "game_id", "homeTeamCode", "awayTeamCode", "isPlayOffGame")
        .groupBy("season", "game_id")
        .agg(
            F.first(F.col("homeTeamCode")).alias("homeTeamCode"),
            F.first(F.col("awayTeamCode")).alias("awayTeamCode"),
            F.first(F.col("isPlayOffGame")).alias("isPlayOffGame"),
            F.lit(0).alias("homeGoals"),
            F.lit(0).alias("awayGoals"),
            F.lit(0).alias("lastGoalTime")
        )
)

gamesDF3: DataFrame = gamesWithGoalsDF.union(gamesWithoutGoalsDF)


# COMMAND ----------

# Final alternative solution, which tries to account for the fact that some goal events are missing from the data.
# For games where the last event is not a goal, but the final goal-scoring event is missing, the "lastGoalTime" is
# set to the time of the event just before the final goal-scoring event should have happened.
gamesDF4: DataFrame = (
    shotsDF
        .withColumn("maxGoals", F.max(F.col("homeTeamGoals") + F.col("awayTeamGoals")).over(gameWindow))
        .withColumn("beforeLastResultChange",
            F.max(
                F.when(F.col("homeTeamGoals") + F.col("awayTeamGoals") < F.col("maxGoals"), F.col("time"))
                    .otherwise(F.lit(0))
            ).over(gameWindow)
        )
        .filter(F.col("homeTeamGoals") + F.col("awayTeamGoals") == F.col("maxGoals"))
        .select(
            F.col("season"),
            F.col("game_id"),
            F.col("homeTeamCode"),
            F.col("awayTeamCode"),
            F.col("isPlayOffGame"),
            F.col("homeTeamGoals"),
            F.col("awayTeamGoals"),
            F.col("beforeLastResultChange"),
            F.when(goalEvent & homeTeam, 1).otherwise(0).alias("isHomeGoal"),
            F.when(goalEvent & awayTeam, 1).otherwise(0).alias("isAwayGoal"),
            F.when(goalEvent, F.col("time")).otherwise(F.col("beforeLastResultChange")).alias("lastGoalTime")
        )
        .groupBy("season", "game_id")
        .agg(
            F.first(F.col("homeTeamCode")).alias("homeTeamCode"),
            F.first(F.col("awayTeamCode")).alias("awayTeamCode"),
            F.first(F.col("isPlayOffGame")).alias("isPlayOffGame"),
            (F.first(F.col("homeTeamGoals")) + F.sum("isHomeGoal")).alias("homeGoals"),
            (F.first(F.col("awayTeamGoals")) + F.sum("isAwayGoal")).alias("awayGoals"),
            F.max("lastGoalTime").alias("lastGoalTime")
        )
)


# COMMAND ----------

# uncomment to produce test output

# print(f"Total number of games: {gamesDF.count()}")
# (
#     gamesDF
#         .filter((F.col("season") == 2011) & (F.col("game_id") == 21188))
#         .union(gamesDF.filter((F.col("season") == 2012) & (F.col("game_id") == 20023)))
#         .union(gamesDF.filter((F.col("season") == 2012) & (F.col("game_id") == 20425)))
#         .union(gamesDF.filter((F.col("season") == 2012) & (F.col("game_id") == 20556)))
#         .union(gamesDF.filter((F.col("season") == 2012) & (F.col("game_id") == 30116)))
#         .union(gamesDF.filter((F.col("season") == 2013) & (F.col("game_id") == 30236)))
#         .union(gamesDF.filter((F.col("season") == 2011) & (F.col("game_id") == 20749)))
#         .union(gamesDF.filter((F.col("season") == 2011) & (F.col("game_id") == 21108)))
#         .show()
# )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 4 - Game wins during playoff seasons
# MAGIC
# MAGIC Create a data frame that uses the game data frame from the basic task 3 and contains aggregated number of wins and losses for each team and for each playoff season, i.e. for games which have been marked as playoff games. Only teams that have played in at least one playoff game in the considered season should be included in the final data frame.
# MAGIC
# MAGIC The following columns should be included in the final data frame:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | season         | integer     | The season for the data. Example: 2009 for the 2009-2010 season |
# MAGIC | teamCode       | string      | The code for the team. For example: TOR, MTL, NYR, etc. |
# MAGIC | games          | integer     | Number of playoff games the team played in the given season |
# MAGIC | wins           | integer     | Number of wins in playoff games the team had in the given season |
# MAGIC | losses         | integer     | Number of losses in playoff games the team had in the given season |
# MAGIC
# MAGIC Playoff games where a team scored more goals than their opponent are considered winning games. And playoff games where a team scored less goals than the opponent are considered losing games.
# MAGIC
# MAGIC In real life there should not be any playoff games where the final score line was even but due to some missing shot data you might end up with a couple of playoff games that seems to have ended up in a draw. These possible "drawn" playoff games can be left out from the win/loss calculations.
# MAGIC

# COMMAND ----------

# a data frame that will have two rows for each game, one for each team
gamesAllDF: DataFrame = (
    gamesDF
        .withColumn(
            "team",
            F.explode(
                F.array(
                    F.struct(
                        F.col("homeTeamCode").alias("code"),
                        F.col("homeGoals").alias("goalsScored"),
                        F.col("awayGoals").alias("goalsConceded")
                    ),
                    F.struct(
                        F.col("awayTeamCode").alias("code"),
                        F.col("awayGoals").alias("goalsScored"),
                        F.col("homeGoals").alias("goalsConceded")
                    )
                )
            )
        )
        .select(
            F.col("season"),
            F.col("game_id"),
            F.col("isPlayOffGame"),
            F.col("team.code").alias("teamCode"),
            F.col("team.goalsScored").alias("goalsScored"),
            F.col("team.goalsConceded").alias("goalsConceded"),
            F.col("lastGoalTime")
        )
        .persist(StorageLevel.MEMORY_ONLY)  # this is used later in this task as well as in basic task 6
)


# COMMAND ----------

# Alternative way to create the same data frame by handling the home and away teams separately and combining the results.
homeGamesDF: DataFrame = (
    gamesDF
        .select(
            F.col("season"),
            F.col("game_id"),
            F.col("isPlayOffGame"),
            F.col("homeTeamCode").alias("teamCode"),
            F.col("homeGoals").alias("goalsScored"),
            F.col("awayGoals").alias("goalsConceded"),
            F.col("lastGoalTime")
        )
)
awayGamesDF: DataFrame = (
    gamesDF
        .select(
            F.col("season"),
            F.col("game_id"),
            F.col("isPlayOffGame"),
            F.col("awayTeamCode").alias("teamCode"),
            F.col("awayGoals").alias("goalsScored"),
            F.col("homeGoals").alias("goalsConceded"),
            F.col("lastGoalTime")
        )
)
gamesAllDF2: DataFrame = (
    homeGamesDF
        .union(awayGamesDF)
        .persist(StorageLevel.MEMORY_ONLY)  #  this would be used later in this task as well as in basic task 6
)


# COMMAND ----------

playoffDF: DataFrame = (
    gamesAllDF
        .filter(F.col("isPlayOffGame") == 1)
        .select(
            F.col("season"),
            F.col("teamCode"),
            F.when(F.col("goalsScored") > F.col("goalsConceded"), 1).otherwise(0).alias("win"),
            F.when(F.col("goalsScored") < F.col("goalsConceded"), 1).otherwise(0).alias("loss")
        )
        .groupBy("season", "teamCode")
        .agg(
            F.count("*").alias("games"),
            F.sum("win").alias("wins"),
            F.sum("loss").alias("losses")
        )
)


# COMMAND ----------

# uncomment to produce test output

# print(f"Total number of rows: {playoffDF.count()}")
# (
#     playoffDF
#         .filter((F.col("season") == 2016) & (F.col("teamCode") == "NSH"))
#         .union(playoffDF.filter((F.col("season") == 2016) & (F.col("teamCode") == "STL")))
#         .union(playoffDF.filter((F.col("season") == 2017) & (F.col("teamCode") == "BOS")))
#         .union(playoffDF.filter((F.col("season") == 2017) & (F.col("teamCode") == "T.B")))
#         .union(playoffDF.filter((F.col("season") == 2011) & (F.col("teamCode") == "NYR")))
#         .union(playoffDF.filter((F.col("season") == 2015) & (F.col("teamCode") == "MIN")))
#         .union(playoffDF.filter((F.col("season") == 2021) & (F.col("teamCode") == "NYR")))
#         .show()
# )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 5 - Best playoff teams
# MAGIC
# MAGIC Using the playoff data frame created in basic task 4 create a data frame containing the win-loss record for best playoff team, i.e. the team with the most wins, for each season. You can assume that there are no ties for the highest amount of wins in each season.
# MAGIC
# MAGIC The following columns should be included in the final data frame:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | season         | integer     | The season for the data. Example: 2009 for the 2009-2010 season |
# MAGIC | teamCode       | string      | The team code for the best performing playoff team in the given season. For example: TOR, MTL, NYR, etc. |
# MAGIC | games          | integer     | Number of playoff games the best performing playoff team played in the given season |
# MAGIC | wins           | integer     | Number of wins in playoff games the best performing playoff team had in the given season |
# MAGIC | losses         | integer     | Number of losses in playoff games the best performing playoff team had in the given season |
# MAGIC
# MAGIC Finally, fetch the details for the best playoff team in season 2022.
# MAGIC

# COMMAND ----------

bestPlayoffTeams: DataFrame = (
    playoffDF
        .withColumn("maxWins", F.max("wins").over(Window.partitionBy("season")))
        .filter(F.col("wins") == F.col("maxWins"))
        .drop("maxWins")
        .sort(F.col("season").asc())
)

bestPlayoffTeams.show()


# COMMAND ----------

bestPlayoffTeam2022: Row = (
    bestPlayoffTeams
        .filter(F.col("season") == 2022)
        .head()
)

bestPlayoffTeam2022Dict: dict = bestPlayoffTeam2022.asDict()
print("Best playoff team in 2022:")
print(f"    Team: {bestPlayoffTeam2022Dict.get('teamCode')}")
print(f"    Games: {bestPlayoffTeam2022Dict.get('games')}")
print(f"    Wins: {bestPlayoffTeam2022Dict.get('wins')}")
print(f"    Losses: {bestPlayoffTeam2022Dict.get('losses')}")
print("=========================================================")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 6 - Regular season points
# MAGIC
# MAGIC Create a data frame that uses the game data frame from the basic task 3 and contains aggregated data for each team and for each season for the regular season matches, i.e. the non-playoff matches.
# MAGIC
# MAGIC The following columns should be included in the final data frame:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | season         | integer     | The season for the data. Example: 2009 for the 2009-2010 season |
# MAGIC | teamCode       | string      | The code for the team. For example: TOR, MTL, NYR, etc. |
# MAGIC | games          | integer     | Number of non-playoff games the team played in the given season |
# MAGIC | wins           | integer     | Number of wins in non-playoff games the team had in the given season |
# MAGIC | losses         | integer     | Number of losses in non-playoff games the team had in the given season |
# MAGIC | goalsScored    | integer     | Total number goals scored by the team in non-playoff games in the given season |
# MAGIC | goalsConceded  | integer     | Total number goals scored against the team in non-playoff games in the given season |
# MAGIC | points         | integer     | Total number of points gathered by the team in non-playoff games in the given season |
# MAGIC
# MAGIC Points from each match are received as follows (in the context of this assignment, these do not exactly match the NHL rules):
# MAGIC
# MAGIC | points | situation |
# MAGIC | ------ | --------- |
# MAGIC | 3      | team scored more goals than the opponent during the regular 60 minutes |
# MAGIC | 2      | the score line was even after 60 minutes but the team scored a winning goal during the extra time |
# MAGIC | 1      | the score line was even after 60 minutes but the opponent scored a winning goal during the extra time or there were no goals in the extra time |
# MAGIC | 0      | the opponent scored more goals than the team during the regular 60 minutes |
# MAGIC
# MAGIC In the regular season the following table shows how wins and losses should be considered (in the context of this assignment):
# MAGIC
# MAGIC | win | loss | situation |
# MAGIC | --- | ---- | --------- |
# MAGIC | Yes | No   | team gained at least 2 points from the match |
# MAGIC | No  | Yes  | team gain at most 1 point from the match |
# MAGIC

# COMMAND ----------

teamPointsDF: DataFrame = (
    gamesAllDF  # home and away teams are already separated in task 4
        .filter(F.col("isPlayOffGame") == 0)
        .withColumn(
            "points",
            F.when((F.col("goalsScored") > F.col("goalsConceded")) & (F.col("lastGoalTime") <= 3600), 3)    # regulation win
                .when((F.col("goalsScored") > F.col("goalsConceded")) & (F.col("lastGoalTime") > 3600), 2)  # overtime win
                .when((F.col("goalsScored") < F.col("goalsConceded")) & (F.col("lastGoalTime") > 3600), 1)  # overtime loss
                .when(F.col("goalsScored") == F.col("goalsConceded"), 1)                                    # no goals in overtime
                .otherwise(0)                                                                               # regulation loss
        )
        .withColumns(
            {
                "win": F.when(F.col("points") >= 2, 1).otherwise(0),
                "loss": F.when(F.col("points") <= 1, 1).otherwise(0)
            }
        )
)


# COMMAND ----------

regularSeasonDF: DataFrame = (
    teamPointsDF
        .groupBy("season", "teamCode")
        .agg(
            F.count("*").alias("games"),
            F.sum("win").alias("wins"),
            F.sum("loss").alias("losses"),
            F.sum("goalsScored").alias("goalsScored"),
            F.sum("goalsConceded").alias("goalsConceded"),
            F.sum("points").alias("points")
        )
)


# COMMAND ----------

# uncomment to produce test output

# print(f"Total number of rows: {regularSeasonDF.count()}")
# (
#     regularSeasonDF
#         .filter((F.col("season") == 2021) & (F.col("teamCode") == "CBJ"))
#         .union(regularSeasonDF.filter((F.col("season") == 2021) & (F.col("teamCode") == "CAR")))
#         .union(regularSeasonDF.filter((F.col("season") == 2015) & (F.col("teamCode") == "COL")))
#         .union(regularSeasonDF.filter((F.col("season") == 2022) & (F.col("teamCode") == "NYI")))
#         .union(regularSeasonDF.filter((F.col("season") == 2016) & (F.col("teamCode") == "MIN")))
#         .union(regularSeasonDF.filter((F.col("season") == 2017) & (F.col("teamCode") == "WPG")))
#         .union(regularSeasonDF.filter((F.col("season") == 2022) & (F.col("teamCode") == "CGY")))
#         .show()
# )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 7 - The worst regular season teams
# MAGIC
# MAGIC Using the regular season data frame created in the basic task 6, create a data frame containing the regular season records for the worst regular season team, i.e. the team with the least amount of points, for each season. You can assume that there are no ties for the lowest amount of points in each season.
# MAGIC
# MAGIC Finally, fetch the details for the worst regular season team in season 2022.
# MAGIC

# COMMAND ----------

worstRegularTeams: DataFrame = (
    regularSeasonDF
        .withColumn("minPoints", F.min("points").over(Window.partitionBy("season")))
        .filter(F.col("points") == F.col("minPoints"))
        .drop("minPoints")
        .sort(F.col("season").asc())
)

worstRegularTeams.show()


# COMMAND ----------

worstRegularTeam2022: Row = (
    worstRegularTeams
        .filter(F.col("season") == 2022)
        .head()
)

worstRegularTeam2022Dict: dict = worstRegularTeam2022.asDict()
print("Worst regular season team in 2022:")
print(f"    Team: {worstRegularTeam2022Dict.get('teamCode')}")
print(f"    Games: {worstRegularTeam2022Dict.get('games')}")
print(f"    Wins: {worstRegularTeam2022Dict.get('wins')}")
print(f"    Losses: {worstRegularTeam2022Dict.get('losses')}")
print(f"    Goals scored: {worstRegularTeam2022Dict.get('goalsScored')}")
print(f"    Goals conceded: {worstRegularTeam2022Dict.get('goalsConceded')}")
print(f"    Points: {worstRegularTeam2022Dict.get('points')}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional tasks
# MAGIC
# MAGIC The implementation of the basic tasks is compulsory for every group.
# MAGIC
# MAGIC Doing the following additional tasks you can gain course points which can help in getting a better grade from the course (or passing the course).
# MAGIC Partial solutions can give partial points.
# MAGIC
# MAGIC The additional task 1 will be considered in the grading for every group based on their solutions for the basic tasks.
# MAGIC
# MAGIC The additional tasks 2 and 3 are separate tasks that do not relate to any other task in the assignment. The solutions used in these other additional tasks do not affect the grading of additional task 1. Instead, a good use of optimized methods can positively affect the grading of each specific task, while very non-optimized solutions can have a negative effect on the task grade.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Task 1 - Optimized solutions to the basic tasks (2 points)
# MAGIC
# MAGIC Use the tools Spark offers effectively and avoid unnecessary operations in the code for the basic tasks.
# MAGIC
# MAGIC A couple of things to consider (**NOT** even close to a complete list):
# MAGIC
# MAGIC - Consider using explicit schemas when dealing with CSV data sources.
# MAGIC - Consider only including those columns from a data source that are actually needed.
# MAGIC - Filter unnecessary rows whenever possible to get smaller datasets.
# MAGIC - Avoid collect or similar expensive operations for large datasets.
# MAGIC - Consider using explicit caching if some data frame is used repeatedly.
# MAGIC - Avoid unnecessary shuffling (for example sorting) operations.
# MAGIC
# MAGIC It is okay to have your own test code that would fall into category of "ineffective usage" or "unnecessary operations" while doing the assignment tasks. However, for the final Moodle submission you should comment out or delete such code (and test that you have not broken anything when doing the final modifications).
# MAGIC
# MAGIC Note, that you should not do the basic tasks again for this additional task, but instead modify your basic task code with more efficient versions.
# MAGIC
# MAGIC You can create a text cell below this one and describe what optimizations you have done. This might help the grader to better recognize how skilled your work with the basic tasks has been.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Spark can in some cases do its own optimizations, and thus some of the used explicit optimizations might not be any more efficient than more straightforward alternatives.
# MAGIC
# MAGIC Also, no extensive testing regarding the solutions has been done. There might well be more optimized ways to do the tasks than what is given in this example.
# MAGIC
# MAGIC Some of the considered optimizations are listed here:
# MAGIC
# MAGIC - Basic task 1
# MAGIC     - Explicit schema is defined for the data to avoid additional reading for the CSV file.
# MAGIC     - The initial data frame is limited to the necessary rows and columns.
# MAGIC     - Even though the initial data frame is used twice, the caching statement is commented out since it did not seem to improve the execution speed.
# MAGIC     - When finding the asked publisher, sorting is avoided by using the max_by function (reduce would be another alternative).
# MAGIC - Basic task 2
# MAGIC     - Columns that are not used in the following tasks are immediately dropped from the data frame.
# MAGIC     - Note that for Parquet data there is no need to provide explicit schema for efficiency since the file itself will contain the schema.
# MAGIC - Basic task 3
# MAGIC     - There are a total of four alternative solutions given. A normal assignment submission would, of course, contain only one.
# MAGIC     - The idea in each alternative is to avoid possible expensive join operations.
# MAGIC     - As for the resulting data frame, the main example outputs have been created with the last alternative (`gamesDF4`). The first alternative (`gamesDF`) will give almost the same results excepts for a couple of games. The other alternatives (`gamesDF2` and `gamesDF3`) will give output that should match the given alternative example outputs.
# MAGIC - Basic task 4
# MAGIC     - There a two alternative solutions given. Again, a normal assignment submission would contain only one.
# MAGIC     - Multiple grouping operations (as well as join operations) are avoided by first creating a data frame with two rows for each game, one for the home team and one for the away team. For the alternative that handles these separately, the combined data frame is created by union. The necessary grouping operation is then done for the combined data frame, from which only the playoff games are included.
# MAGIC     - The combined data frame before the grouping operation is stored into cache. This can then be used as the starting point in basic task 6 without reading anything from disk.
# MAGIC         - For the second alternative (`gamesAllDF2`) also storing the output from basic task 3 (`gamesDF`) into cache could be considered.
# MAGIC - Basic task 5
# MAGIC     - A full sorting is avoided by finding the maximum number of wins for each season using a window functionality.
# MAGIC - Basic task 6
# MAGIC     - Here the code for calculating the number of points for each game could likely be done more efficiently, but it is left as it is to provide better code readability (which is important as well).
# MAGIC     - Otherwise, the task is very similar to the second half of basic task 4. This time, only the regular season games are included from the cached data frame from task 4.
# MAGIC - Basic task 7
# MAGIC     - Very similar to basic task 5, except using the minimum number of points instead of the maximum number of wins.
# MAGIC
# MAGIC Note, that in the assignment grading, any logic fails with the basic tasks in addition to inefficient solutions might have lowered the total points for this task.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Task 2 - Unstructured data (2 points)
# MAGIC
# MAGIC You are given some text files with contents from a few thousand random articles both in English and Finnish from Wikipedia. Content from English articles are in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/wikipedia/en` and content from Finnish articles are at folder `assignment/wikipedia/fi`.
# MAGIC
# MAGIC Some cleaning operations have already been done to the texts but the some further cleaning is still required.
# MAGIC
# MAGIC The final goal of the task is to get the answers to following questions:
# MAGIC
# MAGIC - What are the ten most common English words that appear in the English articles?
# MAGIC - What are the five most common 5-letter Finnish words that appear in the Finnish articles?
# MAGIC - What is the longest word that appears at least 150 times in the articles?
# MAGIC - What is the average English word length for the words appearing in the English articles?
# MAGIC - What is the average Finnish word length for the words appearing in the Finnish articles?
# MAGIC
# MAGIC For a word to be included in the calculations, it should fulfill the following requirements:
# MAGIC
# MAGIC - Capitalization is to be ignored. I.e., words "English", "ENGLISH", and "english" are all to be considered as the same word "english".
# MAGIC - An English word should only contain the 26 letters from the alphabet of Modern English. Only exception is that punctuation marks, i.e. hyphens `-`, are allowed in the middle of the words as long as there are no two consecutive punctuation marks without any letters between them.
# MAGIC - The only allowed 1-letter English words are `a` and `i`.
# MAGIC - A Finnish word should follow the same rules as English words, except that three additional letters, `å`, `ä`, and `ö`, are also allowed, and that no 1-letter words are allowed. Also, any word that contains "`wiki`" should not be considered as Finnish word.
# MAGIC
# MAGIC Some hints:
# MAGIC
# MAGIC - Using an RDD or a Dataset (in Scala) might make the data cleaning and word determination easier than using DataFrames.
# MAGIC - It can be assumed that in the source data each word in the same line is separated by at least one white space (` `).
# MAGIC - You are allowed to remove all non-allowed characters from the source data at the beginning of the cleaning process.
# MAGIC - It is advisable to first create a DataFrame/Dataset/RDD that contains the found words, their language, and the number of times those words appeared in the articles. This can then be used as the starting point when determining the answers to the given questions.
# MAGIC

# COMMAND ----------

# some constants that could be useful
englishLetters: str = "abcdefghijklmnopqrstuvwxyz"
finnishLetters: str = englishLetters + "åäö"
whiteSpace: str = " "
punctuationMark: str = '-'
twoPunctuationMarks: str = "--"
allowedEnglishOneLetterWords: List[str] = ["a", "i"]
wikiStr: str = "wiki"

englishStr: str = "English"
finnishStr: str = "Finnish"


# COMMAND ----------

# the alternative output values can be gotten by setting this to true
strictRules: bool = False

# a data class for holding the word validation rules for a language
@dataclass(frozen=True)
class LanguageRules:
    language: str
    letters: str
    allowedOneLetterWords: List[str]
    prohibitedWord: Optional[str]

    @property
    def allLetters(self) -> str:
        return self.letters + self.letters.upper()

englishRules: LanguageRules = LanguageRules(englishStr, englishLetters, allowedEnglishOneLetterWords, None)
finnishRules: LanguageRules = LanguageRules(finnishStr, finnishLetters, [], wikiStr)


# COMMAND ----------

# Validation check for a single character.
def isAllowedCharacter(c: str, languageRules: LanguageRules) -> bool:
    return c == punctuationMark or c in languageRules.allLetters

# Strip the words from non-allowed characters.
# Different behaviour depending on the value of strictRules variable.
def stripWord(word: str, languageRules: LanguageRules) -> str:
    def containsOnlyAllowedCharacters(testWord: str) -> bool:
        return (
            len(testWord) == 0 or
            (isAllowedCharacter(testWord[0], languageRules) and containsOnlyAllowedCharacters(testWord[1:]))
        )

    if not strictRules:
        # remove all non-allowed characters
        return "".join([letter for letter in word if isAllowedCharacter(letter, languageRules)])

    # return the original word if it contains only allowed characters, otherwise return an empty string
    if containsOnlyAllowedCharacters(word):
        return word
    return ""

# Validation check for a single word.
def isAllowedWord(word: str, languageRules: LanguageRules) -> bool:
    # General checks for the word
    # Here it is assumed that all characters in the word are allowed characters
    return (
        len(word) > 0 and
        word[0] != punctuationMark and
        word[-1] != punctuationMark and
        twoPunctuationMarks not in word and
        (languageRules.prohibitedWord is None or languageRules.prohibitedWord not in word) and
        (len(word) > 1 or word in languageRules.allowedOneLetterWords)
    )

# Returns a data frame with validated words.
def getWordDF(textRDD: RDD[str], languageRules: LanguageRules) -> DataFrame:
    return (
        textRDD
            .flatMap(lambda line: line.split(whiteSpace))             # split the lines of text into words
            .map(lambda word: stripWord(word, languageRules))         # ensure that there are only allowed characters in the word
            .map(lambda word: word.lower())                           # all letters to lower case letters
            .filter(lambda word: isAllowedWord(word, languageRules))  # only keep words with allowed structure
            .map(lambda word: (word, ))                               # convert word strings to tuples
            .toDF(["word"])                                           # convert to data frame
            .withColumn("language", F.lit(languageRules.language))    # add the language column
    )

# Note that here the word validation is done code readability in mind, not to be as efficient as possible during the execution.


# COMMAND ----------

rawDataEn: RDD[str] = spark.sparkContext.textFile("abfss://shared@tunics320f2023gen2.dfs.core.windows.net/assignment/wikipedia/en/*")
rawDataFi: RDD[str] = spark.sparkContext.textFile("abfss://shared@tunics320f2023gen2.dfs.core.windows.net/assignment/wikipedia/fi/*")

# combine the validated English and Finnish words with word counts and lengths into one data frame and cache it
wordDF: DataFrame = (
    getWordDF(rawDataEn, englishRules)
        .union(getWordDF(rawDataFi, finnishRules))
        .groupBy("language", "word")
        .agg(F.count("*").alias("count"))
        .withColumn("word_length", F.length(F.col("word")))
        .persist(StorageLevel.MEMORY_ONLY)  # storing this data frame in memory can provide significant speedup for the following tasks
)


# COMMAND ----------

# uncomment to produce test output

# print(f'English words: {wordDF.filter(F.col("language") == englishStr).select("count").agg(F.sum(F.col("count"))).head()[0]}')
# print(f'English distinct words: {wordDF.filter(F.col("language") == englishStr).select("word").distinct().count()}')
# print(f'Finnish words: {wordDF.filter(F.col("language") == finnishStr).select("count").agg(F.sum(F.col("count"))).head()[0]}')
# print(f'Finnish distinct words: {wordDF.filter(F.col("language") == finnishStr).select("word").distinct().count()}')


# COMMAND ----------

commonWordsEn: DataFrame = (
    wordDF
        .filter(F.col("language") == englishStr)
        .drop("language", "word_length")
        .select("word", "count")
        .orderBy(F.col("count").desc())
        .limit(10)
)

print("The ten most common English words that appear in the English articles:")
commonWordsEn.show()


# COMMAND ----------

common5LetterWordsFi: DataFrame = (
    wordDF
        .filter((F.col("language") == finnishStr) & (F.col("word_length") == 5))
        .select("word", "count")
        .orderBy(F.col("count").desc())
        .limit(5)
)

print("The five most common 5-letter Finnish words that appear in the Finnish articles:")
common5LetterWordsFi.show()


# COMMAND ----------

# Combine the English and Finnish words in case some words appear in both languages, and then find the longest word
# (combining was not required, and just checking the word counts for languages separately was allowed)
longestWord: str = (
    wordDF
        .drop("language")
        .filter(F.col("count") >= 75)  # the word must appear at least 75 times in one of the languages
        .groupBy("word")
        .agg(
            F.first(F.col("word_length")).alias("word_length"),
            F.sum("count").alias("count")
        )
        .filter(F.col("count") >= 150)
        .select(F.max_by(F.col("word"), F.col("word_length")))
        .head()
        [0]
)

print(f"The longest word appearing at least 150 times is '{longestWord}'")


# the average word lengths for the entire dataset (not just for the distinct words)
averageWordLengths: DataFrame = (
    wordDF
        .groupBy("language")
        .agg(
            F.sum(F.col("word_length") * F.col("count")).alias("total_letters"),
            F.sum(F.col("count")).alias("total_words")
        )
        .withColumn("average_word_length", F.round(F.col("total_letters") / F.col("total_words"), 2))
        .drop("total_letters", "total_words")
)

print("The average word lengths:")
averageWordLengths.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Task 3 - K-Means clustering (2 points)
# MAGIC
# MAGIC You are given a dataset containing the locations of building in Finland. The dataset is a subset from [https://www.avoindata.fi/data/en_GB/dataset/postcodes/resource/3c277957-9b25-403d-b160-b61fdb47002f](https://www.avoindata.fi/data/en_GB/dataset/postcodes/resource/3c277957-9b25-403d-b160-b61fdb47002f) limited to only postal codes with the first two numbers in the interval 30-44 ([postal codes in Finland](https://www.posti.fi/en/zip-code-search/postal-codes-in-finland)). The dataset is in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2023-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2023gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/buildings.parquet`.
# MAGIC
# MAGIC [K-Means clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm is an unsupervised machine learning algorithm that can be used to partition the input data into k clusters. Your task is to use the Spark ML library and its K-Means clusterization algorithm to divide the buildings into clusters using the building coordinates `latitude_wgs84` and `longitude_wgs84` as the basis of the clusterization. You should implement the following procedure:
# MAGIC
# MAGIC 1. Start with all the buildings in the dataset.
# MAGIC 2. Divide the buildings into seven clusters with K-Means algorithm using `k=7` and the longitude and latitude of the buildings.
# MAGIC 3. Find the cluster to which the Sähkötalo building from the Hervanta campus is sorted into. The building id for Sähkötalo in the dataset is `102363858X`.
# MAGIC 4. Choose all the buildings from the cluster with the Sähkötalo building.
# MAGIC 5. Find the cluster center for the chosen cluster of buildings.
# MAGIC 6. Calculate the largest distance from a building in the chosen cluster to the chosen cluster center. You are given a function `haversine` that you can use to calculate the distance between two points using the latitude and longitude of the points.
# MAGIC 7. While the largest distance from a building in the considered cluster to the cluster center is larger than 3 kilometers run the K-Means algorithm again using the following substeps.
# MAGIC     - Run the K-Means algorithm to divide the remaining buildings into smaller clusters. The number of the new clusters should be one less than in the previous run of the algorithm (but should always be at least two). I.e., the sequence of `k` values starting from the second run should be 6, 5, 4, 3, 2, 2, ...
# MAGIC     - After using the algorithm again, choose the new smaller cluster of buildings so that it includes the Sähkötalo building.
# MAGIC     - Find the center of this cluster and calculate the largest distance from a building in this cluster to its center.
# MAGIC
# MAGIC As the result of this process, you should get a cluster of buildings that includes the Sähkötalo building and in which all buildings are within 3 kilometers of the cluster center.
# MAGIC
# MAGIC Using the final cluster, find the answers to the following questions:
# MAGIC
# MAGIC - How many buildings in total are in the final cluster?
# MAGIC - How many Hervanta buildings are in this final cluster? (A building is considered to be in Hervanta if their postal code is `33720`)
# MAGIC
# MAGIC Some hints:
# MAGIC
# MAGIC - Once you have trained a KMeansModel, the coordinates for the cluster centers, and the cluster indexes for individual buildings can be accessed through the model object (`clusterCenters`, `summary.predictions`).
# MAGIC - The given haversine function for calculating distances can be used with data frames if you turn it into an user defined function.
# MAGIC

# COMMAND ----------

# some helpful constants
startK: int = 7
seedValue: int = 1

# the building id for Sähkötalo building at Hervanta campus
hervantaBuildingId: str = "102363858X"
hervantaPostalCode: int = 33720

maxAllowedClusterDistance: float = 3.0

# returns the distance between points (lat1, lon1) and (lat2, lon2) in kilometers
# based on https://community.esri.com/t5/coordinate-reference-systems-blog/distance-on-a-sphere-the-haversine-formula/ba-p/902128
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R: float = 6378.1  # radius of Earth in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    deltaPhi = math.radians(lat2 - lat1)
    deltaLambda = math.radians(lon2 - lon1)

    a = (
        math.sin(deltaPhi * deltaPhi / 4.0) +
        math.cos(phi1) * math.cos(phi2) * math.sin(deltaLambda * deltaLambda / 4.0)
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# the distance calculation function as UDF so it can be used with spark
haversineUDF: UserDefinedFunction = F.udf(
    f=lambda lat1, lon1, lat2, lon2: haversine(lat1, lon1, lat2, lon2),
    returnType=DoubleType()
)
# the same could be also defined with a short cut:
# haversineUDF: UserDefinedFunction = F.udf(haversine, DoubleType())


# COMMAND ----------

# objects for the K-Means
assembler: VectorAssembler = (
    VectorAssembler()
        .setInputCols(["latitude_wgs84", "longitude_wgs84"])
        .setOutputCol("features")
)

kMeansBase: KMeans = (
    KMeans()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setSeed(seedValue)
        .setK(startK)
)

# assembled building data frame
buildingDF: DataFrame = (
    spark
        .read
        .parquet("abfss://shared@tunics320f2023gen2.dfs.core.windows.net/assignment/buildings.parquet")
        .select("building_id", "postal_code", "latitude_wgs84", "longitude_wgs84")
        .transform(lambda data: assembler.transform(data))
)


# COMMAND ----------

# recursive function for the K-Means iterations
def iterateKMeans(data: DataFrame, k: int, iteration: int) -> DataFrame:
    model: KMeansModel = (
        kMeansBase
            .setK(k)
            .fit(data)
    )

    clusterCenters: list = model.clusterCenters()
    clusterSizes: List[int] = model.summary.clusterSizes
    predictions: DataFrame = (
        model.summary.predictions
            .persist(StorageLevel.MEMORY_ONLY)  # using cache here does seem to speed up the process
    )

    # cluster index for the cluster where "Sähkötalo" is included
    clusterIndex: int = (
        predictions
            .filter(F.col("building_id") == hervantaBuildingId)
            .select("prediction")
            .first()
            [0]
    )

    # coordinates for the "Sähkötalo" cluster center
    clusterCenterLatitude: float = clusterCenters[clusterIndex][0]
    clusterCenterLongitude: float = clusterCenters[clusterIndex][1]

    # data frame for the "Sähkötalo" cluster
    selectedCluster: DataFrame = (
        predictions
            .filter(F.col("prediction") == clusterIndex)
            .drop("prediction")
    )

    # maximum distance from any building in the "Sähkötalo" cluster to the cluster center
    maxClusterDistance: float = (
        selectedCluster
            .withColumn(
                "distance",
                haversineUDF(
                    F.col("latitude_wgs84"),
                    F.col("longitude_wgs84"),
                    F.lit(clusterCenterLatitude),
                    F.lit(clusterCenterLongitude)
                )
            )
            .agg(F.max("distance").alias("max_distance"))
            .first()
            .asDict()
            .get("max_distance")
    )

    print(f"(k={k}, iteration={iteration}) ", end="")
    print(f"Buildings: {sum(clusterSizes)} -> {clusterSizes[clusterIndex]}, ", end="")
    print(f"Maximum distance to the center within 'Sähkötalo' cluster: {round(maxClusterDistance, 2)} km")

    if maxClusterDistance > maxAllowedClusterDistance:
        newK: int = k-1 if k > 2 else k
        return iterateKMeans(selectedCluster, newK, iteration+1)
    else:
        return selectedCluster


# COMMAND ----------

# start the K-Means iterations
finalCluster: DataFrame = iterateKMeans(buildingDF, startK, 1)

clusterBuildingCount: int = finalCluster.count()
clusterHervantaBuildingCount: int = (
    finalCluster
        .filter(F.col("postal_code") == hervantaPostalCode)
        .count()
)

print(f"Buildings in the final cluster: {clusterBuildingCount}")
print(f"Hervanta buildings in the final cluster: {clusterHervantaBuildingCount} ", end="")
print(f"({round(100*clusterHervantaBuildingCount/clusterBuildingCount, 2)}% of all buildings in the final cluster)")
print("===========================================================================================")

