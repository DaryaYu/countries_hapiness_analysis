# countries_hapiness_analysis

The purpose of the project is:
-	to research and investigate different countries happiness rates, 
-	to figure out which variables from datasets influence happiness level in a country, 
-	to build a model, that predicts happiness level in a country based on the variables predefined,
-	to cluster countries based on the happiness level and other variables (GDP per capita variable used as an example).

Data for the project retrieved from ourworldindata.org and databank.worldbank.org/data/home.aspx.

2 datasets were used in the research:
1) 1CountriesSeries file consists of variables: Country, Adjusted net national income per capita (current US$), GDP per capita,  PPP (current international $), Life expectancy at birth, Literacy rate adult total, Military expenditure (% of central government expenditure), People using basic drinking water services (% of population), People using basic sanitation services (% of population), PM2.5 air pollution population exposed to levels exceeding WHO guideline value (% of total), Population total, Refugee population by country or territory of origin, Total alcohol consumption per capita (liters of pure alcohol projected estimates 15+ years of age), Unemployment, Urban population (% of total). Data is consolidated by country. Data shows 2016 or 2015 years results. 
2) 2Indicators file consists of variables: country, year (2016 only), Hapiness perception, Log GDP per capita, Healthy life expectancy at birth, Freedom to make life choices, Social support, Perceptions of corruption, Confidence in national government. Data is consolidated by country.

Algorithms & methods used in the project:
- KNN and Polinomial Regression for missing values prediction based on the variables dependency visualisation;
- Multiple Linear Regression using Backward Elimination method to identify, which variables define countries hapiness perception the most;
- Random Forrest Regression, Gradient Boosting Regression and Ada Boost Regression for the main model building;
- Explained Variance score, Mean Absolute error, Mean Squared error, Median Absolute error, R2 score to choose the model with the best prediction level; 
- K-means Clustering for classification of countries into groups based the happiness perception level and GDP per capita.
