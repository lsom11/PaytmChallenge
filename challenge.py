import pyspark.sql.functions as F

from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import session, context
from pyspark.sql.window import Window

if __name__ == '__main__':
    conf = SparkConf()
    conf.setMaster("local").setAppName("Paytm_Challenge")
    sc = SparkContext.getOrCreate(conf=conf)
    spark = session.SparkSession(sc)

    Logger = spark._jvm.org.apache.log4j.Logger
    local_logger = Logger.getLogger(__name__)

    #==========================================================================================
    # STEP ONE
    #==========================================================================================

    #==========================================================================================
    ## 1. Load the global weather data into your big data technology of choice.
    #==========================================================================================

    station_df = spark.read.format("csv").option("header", "true").option("inferschema", "true").load("./stationlist.csv")
    country_df = spark.read.format("csv").option("header", "true").option("inferschema", "true").load("./countrylist.csv")

    #==========================================================================================
    ## 2. Join the stationlist.csv with the countrylist.csv to get the full country name for each station number.
    #==========================================================================================

    station_with_country_df = station_df \
                                .join(country_df, station_df.COUNTRY_ABBR == country_df.COUNTRY_ABBR) \
                                .drop(country_df.COUNTRY_ABBR)

    fpath5 = './data/2019/part-00000-890686c0-c142-4c69-a744-dfdc9eca7df4-c000.csv.gz'
    fpath3 = './data/2019/part-00001-890686c0-c142-4c69-a744-dfdc9eca7df4-c000.csv.gz'
    fpath4 = './data/2019/part-00002-890686c0-c142-4c69-a744-dfdc9eca7df4-c000.csv.gz'
    fpath2 = './data/2019/part-00003-890686c0-c142-4c69-a744-dfdc9eca7df4-c000.csv.gz'
    fpath1 = './data/2019/part-00004-890686c0-c142-4c69-a744-dfdc9eca7df4-c000.csv.gz'
    weather_df = spark.read.option("inferschema", "true").csv([fpath1, fpath2, fpath3, fpath4, fpath5], header=True)

    #==========================================================================================
    ## 3. Join the global weather data with the full country names by station number.
    #==========================================================================================

    weather_with_station_df = weather_df \
                            .join(station_with_country_df, weather_df['STN---'] == station_with_country_df.STN_NO) \
                            .drop(weather_df['STN---']) \
                            .withColumn('YEAR', weather_df['YEARMODA'].substr(1,4))

    #==========================================================================================
    ## STEP TWO
    #==========================================================================================

    #==========================================================================================
    # 1. Which country had the hottest average mean temperature over the year?
    #==========================================================================================

    hottest_avg_mean_temp = weather_with_station_df\
                            .filter(weather_df.TEMP != 999.9)\
                            .groupBy('COUNTRY_FULL', 'YEAR')\
                            .agg(F.mean("TEMP")\
                            .alias("average_temp_by_country"))\
                            .orderBy("average_temp_by_country", ascending=False).head()

    local_logger.info(f"The country with the hottest average mean temperature is: {hottest_avg_mean_temp}")
    ### After aggregating by the country as well as year (although there is only data for 2019 I decided to group by year as a best practice in case there were other years in the Dataset)
    ### Djibouti is the country with the hottest average mean temperature.

    # ==========================================================================================
    # 2. Which country had the most consecutive days of tornadoes/funnel cloud formations?
    #==========================================================================================

    # We only need the FRSHTT for the filter, so we check the sixth digit for a 1 and discard the rest
    filtered_tornado_df = weather_with_station_df.filter(weather_with_station_df.FRSHTT.substr(-1,1) == '1')

    # Use lag to check if the difference between the last and current row is one (which means it is consecutive) and give it a rank
    country_prev_day_diff = filtered_tornado_df \
                            .withColumn('prev_day_diff', filtered_tornado_df.YEARMODA - F.lag(filtered_tornado_df.YEARMODA) \
                            .over(Window.partitionBy(filtered_tornado_df["COUNTRY_FULL"]) \
                            .orderBy(filtered_tornado_df["YEARMODA"])))

    # Get the sum of all consecutive days if the diff is 1 otherwise ignore
    countries_with_consecutive_warning_days = country_prev_day_diff \
                            .groupBy(country_prev_day_diff.COUNTRY_FULL) \
                            .agg(F.sum(F.when(country_prev_day_diff.prev_day_diff==1,1).otherwise(0)) \
                            .alias('consecutive_days'))

    # Get first country of order by
    countries_with_consecutive_warning_days = countries_with_consecutive_warning_days \
                                                .orderBy("consecutive_days", ascending = False).head()

    local_logger.info(f"The country with the most consecutive tornado or funnel cloud warning days is: {countries_with_consecutive_warning_days}")

    ### Italy is the country with the most consecutive tornado or funnel cloud warning days

    #==========================================================================================
    # 3. Which country had the second highest average mean wind speed over the year?
    #==========================================================================================

    second_windiest_avg_mean_speed = weather_with_station_df\
                          .filter(weather_df.WDSP != 999.9)\
                          .groupBy('COUNTRY_FULL', 'YEAR')\
                          .agg(F.mean("WDSP")\
                          .alias("average_wind_speed_by_country"))\
                          .orderBy("average_wind_speed_by_country", ascending=False).collect()[1]

    local_logger.info(f"the country with the second highest average mean wind speed over the year is: {second_windiest_avg_mean_speed}")
    second_windiest_avg_mean_speed
    ### Bermuda is the country with the second highest average mean wind speed over the year
