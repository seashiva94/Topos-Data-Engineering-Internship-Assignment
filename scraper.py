import requestsx
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

base_url= 'https://en.wikipedia.org/wiki/'
url = base_url + 'List_of_United_States_cities_by_population'
res = requests.get(url)
soup = BeautifulSoup(res.content,'xml')
table = soup.find_all('table')[4]
cities_df = pd.read_html(str(table))[0]
cities_df.columns = ['2018rank', 'city', 'state', '2018estimate', '2010census', 'change(%)', '2016 land area(sq mi)', 
                     '2016 land area(sq km)', '2016 population density (/sq mi)', '2016 population density (/sq km)',
                    'location']

cities_df.drop(0, axis = 0, inplace=True)
print(cities_df.head())

cities_df.city = cities_df.city.apply(lambda x: x.split('[')[0])

city_url = cities_df.city.apply(lambda x: base_url + "_".join(x.split()))
cities_df['wiki_city_url'] = city_url

def get_weather_data(city, state, code):
    url='https://www.usclimatedata.com/climate/' + city + '/' + state + '/united-states/' + code
    table_idx = 2
    res = requests.get(url)
    soup = BeautifulSoup(res.content)
    table = soup.find_all('table')[table_idx]
    df = pd.read_html(str(table))[0]
    df = df.T.dropna(axis = 1)
    df.columns = df.iloc[0,:]
    df = df.drop(0, axis = 0)
    return df

all_states = 'https://www.usclimatedata.com/'
state_table = BeautifulSoup(requests.get(all_states).content).find_all('table')[3]
state_urls = state_table.findAll('a')
test[-2].attrs['href']

state_codes = {}
for entry in state_urls:
    url = entry.attrs['href']
    state = url.split('/')[2]
    code = url.split('/')[-1]
    state_codes[state] = code

city_codes = {}

for state in state_codes.keys():
    state_city_codes = {}
    url='https://www.usclimatedata.com/climate/'
    city_table = BeautifulSoup(requests.get(url + state + 
                                            '/united-states/' + state_codes[state])\
                               .content).find_all('table')[3]
    
    city_urls = city_table.findAll('a')
    for entry in city_urls:
        url = entry.attrs['href']
        city = url.split('/')[2]
        code = url.split('/')[-1]
        state_city_codes[city] = code
    if state == "new-york":
        print(state_city_codes)
    city_codes[state] = state_city_codes


weather_cols = None
for i in range(cities_df.shape[0]):
    city = '-'.join(cities_df.iloc[i].city.lower().split())
    state = '-'.join(cities_df.iloc[i].state.lower().split())
    code = city_codes[state].get(city)
    data = ['-']*7
    if code:
        data = get_weather_data(city=city, state = state, code = code).values
    if weather_cols is not None:
        weather_cols = np.vstack((weather_cols, data))
    else:
        weather_cols = data


weather_col_names = ["Annual high temperature(F)", "Annual low temperature(F)", "Average temperature(F)", 
                    "Average annual precipitation - rainfall(in)", "Days per year with precipitation - rainfall(days)", 
                    "Annual hours of sunshine (hours)", "Av. annual snowfall"]

len(weather_cols), len(weather_cols[0])

weather_df = pd.DataFrame(np.array(weather_cols))
weather_df.columns = weather_col_names
print(weather_df.head())
final_cols = np.hstack((cities_df.columns , weather_df.columns))
final_data = np.hstack((cities_df.values, weather_df.values))

final_df = pd.DataFrame(final_data)
final_df.columns = final_cols
final_df['2016 land area(sq km)'] = final_df['2016 land area(sq km)'].apply(lambda x: str(x).split()[0])
final_df['2016 land area(sq km)'] = final_df['2016 land area(sq km)'].apply(lambda x: "".join(x.split(",")))

final_df['2016 land area(sq mi)'] = final_df['2016 land area(sq mi)'].apply(lambda x: str(x).split()[0])
final_df['2016 land area(sq mi)'] = final_df['2016 land area(sq mi)'].apply(lambda x: "".join(x.split(",")))
final_df['2016 population density (/sq km)'] = final_df['2016 population density (/sq km)'].apply(lambda x: str(x).split('/')[0])
final_df['2016 population density (/sq km)'] = final_df['2016 population density (/sq km)'].apply(lambda x: "".join(x.split(",")))

final_df['2016 population density (/sq mi)'] = final_df['2016 population density (/sq mi)'].apply(lambda x: str(x).split('/')[0])
final_df['2016 population density (/sq mi)'] = final_df['2016 population density (/sq mi)'].apply(lambda x: "".join(x.split(",")))


final_df['Annual high temperature(F)'] = final_df['Annual high temperature(F)'].apply(lambda x:x.split('°')[0])
final_df['Annual low temperature(F)'] = final_df['Annual low temperature(F)'].apply(lambda x:x.split('°')[0])
final_df['Average temperature(F)'] = final_df['Average temperature(F)'].apply(lambda x:x.split('°')[0])
final_df['Average annual precipitation - rainfall(in)'] = final_df['Average annual precipitation - rainfall(in)'].apply(lambda x: x.split()[0])

final_df['Annual hours of sunshine (hours)'] = final_df['Annual hours of sunshine (hours)'].apply(lambda x:x.split()[0])
final_df['Days per year with precipitation - rainfall(days)'] = final_df['Days per year with precipitation - rainfall(days)'].apply(lambda  x: x.split()[0])

final_df['change(%)'] = final_df['change(%)'].apply(lambda x: str(x).split('%')[0])
final_df['change(%)'] = final_df['change(%)'].apply(lambda x: '-' + x[1:] if x[0] is not '+' else '+' + x[1:])


print(final_df.head())

final_df.replace(to_replace='-', value=np.nan, inplace=True)
final_df.replace(to_replace='-ab]', value=0, inplace=True)

final_df['2018rank'] = final_df['2018rank'].astype(np.int)
final_df.city = final_df.city.astype(str)
final_df.state = final_df.state.astype(str)
final_df['2018estimate'] = final_df['2018estimate'].astype(np.int)
final_df['2010census'] = final_df['2010census'].astype(np.int)
final_df['2016 land area(sq km)'] = final_df['2016 land area(sq km)'].astype(np.float)
final_df['2016 land area(sq mi)'] = final_df['2016 land area(sq mi)'].astype(np.float)
final_df['2016 population density (/sq km)'] = final_df['2016 population density (/sq km)'].astype(np.float)
final_df['2016 population density (/sq mi)'] = final_df['2016 population density (/sq mi)'].astype(np.float)
final_df['2016 population density (/sq mi)'] = final_df['2016 population density (/sq mi)'].astype(np.float)

final_df['Annual high temperature(F)'] = final_df['Annual high temperature(F)'].astype(np.float)
final_df['Annual low temperature(F)'] = final_df['Annual low temperature(F)'].astype(np.float)
final_df['Average temperature(F)'] = final_df['Average temperature(F)'].astype(np.float)
final_df['Average annual precipitation - rainfall(in)'] = final_df['Average annual precipitation - rainfall(in)'].astype(np.float)

final_df['Annual hours of sunshine (hours)'] = final_df['Annual hours of sunshine (hours)'].astype(np.float)
final_df['Days per year with precipitation - rainfall(days)'] = final_df['Days per year with precipitation - rainfall(days)'].astype(np.float)

final_df['change(%)'] = final_df['change(%)'].astype(float)
print(final_df.dtypes)


# In[327]:


print(final_df['Av. annual snowfall'].isna().sum())
print(final_df['Annual hours of sunshine (hours)'].isna().sum())
print(final_df['Days per year with precipitation - rainfall(days)'].isna().sum())

final_df.drop(['Av. annual snowfall', 'Annual hours of sunshine (hours)','Days per year with precipitation - rainfall(days)'],\
              axis = 1, inplace = True)

print(final_df.head())

final_df.to_csv(path_or_buf="cities_data.csv", header=True, sep=',', na_rep="-")

### EDA
if EDA ==  True:
    import seaborn as sns
    import matplotlib.pyplot as plt

    final_df.columns
    eda_cols = ['2018estimate', '2010census', 'change(%)', '2016 land area(sq mi)','2016 population density (/sq mi)']


    eda_df = final_df.replace(to_replace=np.nan, value="0")
    sns.pairplot(eda_df, vars=eda_cols )


    temp_cols = ['Annual high temperature(F)','Annual low temperature(F)', 'Average temperature(F)','Average annual precipitation - rainfall(in)']

    sns.pairplot(data=final_df, vars=temp_cols, range=(0, 100))


    plt.scatter(final_df[temp_cols[2]], final_df[temp_cols[3]])
    plt.xlabel(temp_cols[2])
    plt.ylabel(temp_cols[3])
    plt.savefig("plot 1")

    sns.regplot(final_df[temp_cols[2]], final_df[temp_cols[3]])
    sns.regplot(final_df[temp_cols[0]], final_df[temp_cols[3]])
    sns.regplot(final_df[temp_cols[1]], final_df[temp_cols[3]])
    sns.distplot(final_df['2016 population density (/sq mi)'])
    print(final_df['2016 population density (/sq mi)'].median())
    mean_df = final_df.groupby('state').mean()

    sns.distplot(mean_df['2016 population density (/sq mi)'])
    
    
    sns.distplot(mean_df['Annual high temperature(F)'].dropna())
    
    total_df = final_df.groupby('state').sum()
    sns.distplot(total_df['2018estimate'])
    
    sns.distplot(total_df['2010census'])

