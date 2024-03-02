# Data-Intensive Programming Problems

## Example outputs

This notebook contains some example outputs and additional hints for the assignment tasks.

Your output from the tasks do not have to match these examples exactly. Instead these are provided to help you confirm that you are on the right track with the tasks.

## Basic Task 1

Example output from the task:
- - - - - - - - - - - - - - -

```text
The publisher with the highest total video game sales in European Union is: ´Nintendo'
Sales data for the publisher:
+----+--------+------------+
|Year|EU_Total|Global_Total|
+----+--------+------------+
|2000|    6.42|       34.05|
|2001|    8.84|       45.37|
|2002|    9.89|       48.31|
|2003|    7.08|       38.14|
|2004|   12.43|       60.65|
|2005|   42.69|      127.47|
|2006|   60.35|      205.61|
|2007|   32.81|      104.18|
|2008|   25.13|       91.22|
|2009|   36.18|      128.89|
+----+--------+------------+
```

## Basic Task 2

No output is needed from this task.

Some hints about the `shotsDF` data frame:

- There should be `1279180` rows in the data frame.
- The source data contains 15 columns but not all of them will be needed in the other basic tasks. Consider dropping those columns that are not needed.

## Basic Task 3

No output is needed from this task.

Some hints about the task:

- There is data from `15079` games and thus the data frame `gamesDF` should have this many rows.
- It might be easier to handle the games that did not have any goal-scoring shots and those that did have at least one goal-scoring shot separately. The results could then be combined for the final result.
- The source dataset does contain few peculiarities like missing goal-scoring shots or several goal-scoring shots marked for the same second in the same game. These instances are rare and you are allowed to implement your solution without thinking about them. I.e., you are allowed to assume that all source data rows are valid and that there is no missing data.
- Your data frame might have the rows in a different order but at least the following data should be included in `gamesDF`:

```text
+------+-------+------------+------------+-------------+---------+---------+------------+
|season|game_id|homeTeamCode|awayTeamCode|isPlayOffGame|homeGoals|awayGoals|lastGoalTime|
+------+-------+------------+------------+-------------+---------+---------+------------+
|  2011|  21188|         L.A|         EDM|            0|        2|        0|        3448|
|  2012|  20023|         TOR|         BUF|            0|        1|        2|        3498|
|  2012|  20425|         COL|         CHI|            0|        2|        5|        3390|
|  2012|  20556|         COL|         DET|            0|        2|        3|        3884|
|  2012|  30116|         NYI|         PIT|            1|        3|        4|        4069|
|  2013|  30236|         MIN|         CHI|            1|        1|        2|        4182|
|  2011|  20749|         BUF|         NYR|            0|        0|        0|           0|
|  2011|  21108|         L.A|         STL|            0|        0|        0|           0|
+------+-------+------------+------------+-------------+---------+---------+------------+
```

## Basic Task 4

No output is needed from this task.

Some hints about the task:

- While it is possible to do the task with one chain of operations, it might be helpful to do the task in parts. For example, handling the home teams and away teams separately and combining the results for the final data frame.
- The 2019 playoff season had 24 teams and the other 11 considered seasons had 16 playoff teams each. Thus, the final data frame `playoffDF` should have `200` rows. (11\*16+24)
- Your data frame might have the rows in a different order but at least the following data should be included in `playoffDF`:

```text
+------+--------+-----+----+------+
|season|teamCode|games|wins|losses|
+------+--------+-----+----+------+
|  2016|     NSH|   22|  14|     8|
|  2016|     STL|   11|   6|     5|
|  2017|     BOS|   12|   5|     7|
|  2017|     T.B|   17|  11|     6|
|  2011|     NYR|   20|  10|    10|
|  2015|     MIN|    6|   2|     4|
|  2021|     NYR|   20|  10|    10|
+------+--------+-----+----+------+
```

## Basic Task 5

A hint about the task:

- The window function might be helpful in this task: [https://sparkbyexamples.com/spark/spark-sql-window-functions/](https://sparkbyexamples.com/spark/spark-sql-window-functions/)

Example output from the task:
- - - - - - - - - - - - - - -

```text
+------+--------+-----+----+------+
|season|teamCode|games|wins|losses|
+------+--------+-----+----+------+
|  2011|     L.A|   20|  16|     4|
|  2012|     CHI|   23|  16|     7|   <--- here you might have 15 wins depending on how you counted the goals for the games
|  2013|     L.A|   26|  16|    10|
|  2014|     CHI|   23|  16|     7|
|  2015|     PIT|   24|  16|     8|
|  2016|     PIT|   25|  16|     9|
|  2017|     WSH|   24|  16|     8|
|  2018|     STL|   26|  16|    10|
|  2019|     T.B|   25|  18|     7|
|  2020|     T.B|   23|  16|     7|
|  2021|     COL|   20|  16|     4|
|  2022|     VGK|   22|  16|     6|
+------+--------+-----+----+------+
```

and

```text
Best playoff team in 2022:
    Team: VGK
    Games: 22
    Wins: 16
    Losses: 6
```

## Basic Task 6

No output is needed from this task.

Some hints about the task:

- It might be helpful to do the task in parts. For example, first calculating the points from each game, then calculating whether it was win/loss, and finally doing the aggregation operations.
- The seasons 2011-2016 had 30 teams, seasons 2017-2020 had 31 teams, and seasons 2021-2022 had 32 teams. Thus, the final data frame `regularSeasonDF` should have `368` rows. (6\*30+4\*31+2\*32)
- Your data frame might have the rows in a different order but at least the following data should be included in `regularSeasonDF`:

```text
+------+--------+-----+----+------+-----------+-------------+------+
|season|teamCode|games|wins|losses|goalsScored|goalsConceded|points|
+------+--------+-----+----+------+-----------+-------------+------+
|  2021|     CBJ|   82|  33|    49|        258|          297|   103|
|  2021|     CAR|   82|  53|    29|        277|          200|   162|
|  2015|     COL|   82|  35|    47|        212|          240|   111|
|  2022|     NYI|   82|  41|    41|        242|          217|   128|
|  2016|     MIN|   82|  46|    36|        263|          206|   145|
|  2017|     WPG|   82|  48|    34|        273|          216|   153|
|  2022|     CGY|   82|  36|    46|        258|          247|   122|
+------+--------+-----+----+------+-----------+-------------+------+
```

## Basic Task 7

Example output from the task:
- - - - - - - - - - - - - - -

```text
+------+--------+-----+----+------+-----------+-------------+------+
|season|teamCode|games|wins|losses|goalsScored|goalsConceded|points|
+------+--------+-----+----+------+-----------+-------------+------+
|  2011|     CBJ|   82|  25|    57|        198|          257|    84|
|  2012|     FLA|   48|  12|    36|        109|          170|    44|
|  2013|     BUF|   82|  14|    68|        150|          243|    56|
|  2014|     BUF|   82|  15|    67|        153|          269|    60|
|  2015|     TOR|   82|  23|    59|        192|          240|    83|
|  2016|     COL|   82|  21|    61|        165|          276|    61|
|  2017|     BUF|   82|  24|    58|        198|          278|    80|
|  2018|     OTT|   82|  29|    53|        243|          300|    87|
|  2019|     DET|   71|  14|    57|        142|          265|    49|
|  2020|     BUF|   56|  11|    45|        134|          196|    44|
|  2021|     MTL|   82|  19|    63|        218|          317|    68|
|  2022|     ANA|   82|  20|    62|        206|          335|    68|
+------+--------+-----+----+------+-----------+-------------+------+
```

and

```text
Worst regular season team in 2022:
    Team: ANA
    Games: 82
    Wins: 20
    Losses: 62
    Goals scored: 206
    Goals conceded: 335
    Points: 68
```

- - - - - - - - - - - - - - -
A different way of calculating the game results might give the following output for the data frame:

```text
+------+--------+-----+----+------+-----------+-------------+------+
|season|teamCode|games|wins|losses|goalsScored|goalsConceded|points|
+------+--------+-----+----+------+-----------+-------------+------+
|  2011|     CBJ|   82|  25|    57|        197|          257|    84|
|  2012|     FLA|   48|  12|    36|        107|          170|    44|
|  2013|     BUF|   82|  14|    68|        149|          242|    56|
|  2014|     BUF|   82|  15|    67|        153|          268|    60|
|  2015|     TOR|   82|  23|    59|        192|          240|    83|
|  2016|     COL|   82|  21|    61|        165|          276|    61|
|  2017|     BUF|   82|  24|    58|        198|          277|    80|
|  2018|     OTT|   82|  29|    53|        243|          299|    88|
|  2019|     DET|   71|  14|    57|        142|          265|    49|
|  2020|     BUF|   56|  11|    45|        133|          196|    44|
|  2021|     MTL|   82|  19|    63|        216|          316|    68|
|  2022|     ANA|   82|  20|    62|        205|          335|    68|
+------+--------+-----+----+------+-----------+-------------+------+
```

And some other variations could also be possible.

## Additional Task 1

No special hints for this task other than those that have already been given in the task instructions.

## Additional Task 2

*Updated* on 23.11.2023 with some new word count and other output values.

Some hints about the task:

- There should be `728103` lines in the raw text collection of the English articles.
- There should be `714745` lines in the raw text collection of the Finnish articles.
- Your numbers might not be exactly the same as given below but they should give an indication on whether you have understood the word cleaning task in the way the task creator intended:
    - A total of `2423148` words were found from after cleaning from the English articles. From these `114700` were distinct words.
        - With an alternate way to determine the English words, the numbers were `2153257` words with `96340` distinct ones.
    - A total of `1394751` words were found from after cleaning from the Finnish articles. From these `241147` were distinct words.
        - With an alternate way to determine the Finnish words, the numbers were `1226734` words with `205992` distinct ones.
    - The number of distinct words indicate that a lot "made-up" words are included in the count. However, in this task any further cleaning is not necessary.

Example output from the task (possible alternative values given on the right, they would not be part of the normal output):
- - - - - - - - - - - - - - -

```text
The ten most common English words that appear in the English articles:
+----+------+
|word| count|             alternative counts
+----+------+
| the|161631|                    160326
|  of| 85770|                     85631
| and| 68132|                     67710
|  in| 62938|                     62056
|  to| 48409|                     48227
|   a| 43486|                     42126
| was| 22860|                     22767
|  by| 18914|                     18810
|  as| 18684|                     18449
| for| 17182|                     16928
+----+------+
```

```text
The five most common 5-letter Finnish words that appear in the Finnish articles:
+-----+-----+
| word|count|             alternative counts
+-----+-----+
|mutta| 3567|                      3560
|hänen| 2054|                      2052
|jonka| 1873|                      1865
|jossa| 1635|                      1631
|ollut| 1614|                      1565
+-----+-----+
```

```text
The longest word appearing at least 150 times is 'yhdysvaltalainen'
The average word lengths:
+--------+-------------------+
|language|average_word_length|             alternative average lengths
+--------+-------------------+
| Finnish|               7.85|                          7.75
| English|               5.26|                          5.13
+--------+-------------------+
```

## Additional Task 3

Example output from the task:
- - - - - - - - - - - - - - -

The output from the iterations is not asked for in the task instructions but is given here to give indication whether your solution is going in the right direction. These values are created using the seed value 1 for the K-Means algorithm.

```text
(k=7, iteration=1) Buildings: 343556 -> 84990, Maximum distance to the center within 'Sähkötalo' cluster: 82.68 km
(k=6, iteration=2) Buildings: 84990 -> 25299, Maximum distance to the center within 'Sähkötalo' cluster: 28.03 km
(k=5, iteration=3) Buildings: 25299 -> 10581, Maximum distance to the center within 'Sähkötalo' cluster: 13.61 km
(k=4, iteration=4) Buildings: 10581 -> 2304, Maximum distance to the center within 'Sähkötalo' cluster: 10.94 km
(k=3, iteration=5) Buildings: 2304 -> 809, Maximum distance to the center within 'Sähkötalo' cluster: 9.09 km
(k=2, iteration=6) Buildings: 809 -> 757, Maximum distance to the center within 'Sähkötalo' cluster: 2.75 km
Buildings in the final cluster: 757
Hervanta buildings in the final cluster: 669 (88.38% of all buildings in the final cluster)
```
