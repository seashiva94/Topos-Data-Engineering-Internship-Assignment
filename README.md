### Arnav Arnav Data Engineering Internship Assignment

The data was scrapped from Wikipdia link provided in the instructions
file.  Additional data about the weather for each of the cities were
then scrapped from [here](https://www.usclimatedata.com/climate).

The links for each of the state pages were first used to get the state
code used to the website, then each of the city codes from each of the
states were found to generate the urls and scrape annual weather
statistics for each of the cities in the initial list. If not found
NaN values were assigned.

The scraping was done using BeautifulSoup and requests python modules
and pandas and numpy were used for data cleaning.

Some priliminary Exploraory analysis was done using seaborn and
matplotlib.


To run the scraper and save the data in the csv file run:

./scraper.py -f [filepath]

or simply  run

./scraper.py

to scrape and save the data in "cities_data.csv"

where filename is the relative path to the csv file