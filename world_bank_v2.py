import pandas as pd
from pandas import json_normalize
import plotly.graph_objects as go
import dash 
from dash import dcc
from dash import html
from dash import Input, Output, State
import wbdata as wb
from datetime import date
import wbdata.cache as wb_cache
from wbdata import Client
import os

## To prevent update while loading page
## We want to update only when the user is clicking the submit button
from dash.exceptions import PreventUpdate

# Machine Learning integration
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


app = dash.Dash()

# Set the current working directory
os.chdir("C:/Users/ontov/OneDrive/Desktop/Courses/Plotly and Dash/Python Projects Plotly Dash/Examples/World Bank/v2/")

# Create a Client object with custom caching settings
client = Client()

data_topics = client.get_topics()

# print(client.get_topics())
df = pd.DataFrame(data_topics)

################# FOR TOPIC OPTIONS #################
topic_options =  []

for id in df['id'].values:
    topic_dict = {}
    topic = df[df['id'] == id]['value'].values[0]
    # print(id, topic)
    topic_dict['label'] = topic
    topic_dict['value'] = id
    # print(topic_dict)
    topic_options.append(topic_dict)

# for dict in topic_options:
#     print(dict)

################# FOR COUNTRY OPTIONS #################
country_options = []
countries = wb.get_countries()
# print(countries)
# print(type(countries))
df_countries = json_normalize(countries)
# print(df_countries.head())

for country_id in df_countries['id']:
    country_dict = {}
    country_name = df_countries[df_countries['id'] == country_id]['name'].values[0]
    # print(country_name)
    country_dict['label'] = country_name
    country_dict['value'] = country_id
    # print(country_dict)
    country_options.append(country_dict)

# for dict in country_options:
#     print(dict)

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(id = 'dropdown-topics', value = topic_options[0]['value'], options = topic_options),
        dcc.Dropdown(id = 'dropdown-indicators', value =[], options = [], placeholder='Choose Indicator'),
        dcc.Dropdown(id = 'dropdown-countries', value = [], options = country_options, placeholder='Choose countries', multi=True),
        html.Div([
            html.P(children= 'Minimum Date allowed -> 1960,1,1'),
            html.P(children= "Maximum Date Allowed -> Today's date"),
            dcc.DatePickerRange(id = 'date-picker-range', start_date = date(1960,1,1), end_date= date(2020,1,1), min_date_allowed=date(1960,1,1), max_date_allowed=date.today()),
        ]),
        html.Button(id ='submit-button', n_clicks=0, children='Submit'),
        html.P(id='temp', children='Countries List')
    ]),
    html.Div(
        dcc.Graph(id = 'graph-countries-indicators',
                  figure = {
                      'data': [go.Scatter(
                          x = [],
                          y = [],
                          mode = 'lines'
                      )],
                      'layout': go.Layout(title = 'Countires/Indicators') 
                  })
    )
])

@app.callback(Output('dropdown-indicators', 'options'),
              [Input('dropdown-topics','value')])
def submit_topic(topic_value):
    indicator_options = []
    # print(topic_value) # this is an int that indicates the topic id
    indicators_search= wb.get_indicators(topic = topic_value , skip_cache = True)
    # print(indicators_search) # id --- name (json)
    df_indicators = json_normalize(indicators_search) # type df
    # print(type(df_indicators)) # type dataframe
    
    # print(type(df_indicators)) # dataframe
    # print(df_indicators.head())
    for indicator in df_indicators['id']:
        indicators_dict = {}
        # print(indicator) # indicator ex: TX.VAL.AGRI.ZS.UN
        name = df_indicators[df_indicators['id'] == indicator]['name'].values[0]
        # print(name) # actual name of the indicator
        indicators_dict['label'] = name
        indicators_dict['value'] = indicator
        indicator_options.append(indicators_dict)
    
    # for dict in indicator_options:
    #     print(dict)
    return indicator_options


@app.callback(Output('graph-countries-indicators','figure'),
              [Input('submit-button','n_clicks')],
              [State('dropdown-countries', 'value'),
               State('dropdown-indicators', 'value'),
               State('date-picker-range', 'start_date'),
               State('date-picker-range', 'end_date')])
def update_plot(n_clicks,countries_list,indicator,start_date,end_date):
    # Prevent update without clicking the submti button
    if n_clicks == 0:
        raise PreventUpdate
    else:
        traces = []
        # print(indicator, countries_list)
        # temp_dict = {indicator: 'indicator_values'}
        # print(temp_dict)
        
        # df_plot = wb.get_dataframe(indicators= temp_dict, country = countries_list)
        # print(df_plot.head())
        # print(df_plot.tail())
        # print(df_plot.columns)
        for country in countries_list:
            temp_dict = {indicator: 'indicator_values'}
            df_plot = wb.get_dataframe(indicators= temp_dict, country = country,  date=(str(start_date),str(end_date)))

            


            df_plot.reset_index(inplace=True)
            print("----------------------------------------------------------------")
            print("Country :" + country)
            # print(df_plot.head())
            # print(df_plot.tail())
            # print(df_plot.columns)
            ## Convert date column from str to datetime() elements
            df_plot['date'] = pd.to_datetime(df_plot['date'])
            # Convert datetime to date
            df_plot['date'] = df_plot['date'].dt.date
            # print(type(df_plot['date'].values[0]))
            # print(df_plot['date'].values[0].year)
            # print(type(df_plot['indicator_values']))

            
        

            
            ## REMOVE NAN VALUES
            df_plot.dropna(inplace=True)

            ## After dropping Nan Values we should check if the DataFrame is empty or not
            # If it is empty -> continue to the next country 
            if len(df_plot) == 0:
                print('We should go to the next country at this point')
                continue

            print(len(df_plot))
            print(df_plot.head())
            print(df_plot.tail())
            print(df_plot.columns)
            print("----------------------------------------------------------------")

            
            trace = go.Scatter(
                    x = df_plot['date'],
                    y = df_plot['indicator_values'],
                    mode = 'markers+lines',
                    name = country
                    )
            traces.append(trace) 

            ## Machine Learing session:
            # Assuming df is your DataFrame containing indicator values
            # Ensure DataFrame is sorted by date in descending order

            # Feature Engineering: Create lagged versions of indicator values as features
            df_ml = df_plot
            
            
            ## BASIC: Remove missing values
            ## Remove Nan and None values from our df_ml to prepare it for Machine Learing Session
            ## ADVANCED: Replace values
            ## We could replace missing values with other values based on observations or prediction?

            df_ml = df_ml.dropna()
            
            df_ml['lag1'] = df_plot['indicator_values'].shift(1)  # Lagged by one time step
            print('------------------------------------------------------------------------')
            print('Machine Learing Session')
            print(df_ml.head()) 
            print(df_ml.tail())
            # print('DF_PLOT len: {}'.format(len(df_plot)))
            # print('DF_ML len: {}'.format(len(df_ml)))
            
            # Prepare data for training  
            X = df_ml[['lag1']].dropna()  # Features (use lagged values)
            y = df_ml.loc[X.index, 'indicator_values']  # Target variable ## X.index -> Get every row of X

            ## Standarize data in range of 0 , 1 
  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
            
            model = LinearRegression()
           
            X = X.reset_index(drop = True)
            y = y.reset_index(drop = True)
            # print(X)
            # print(type(X))
            # print(len(X))
            # print(X.columns)
            

            # print(y)
            # print(type(y))
            # print(len(y))
            
            # print(X_train)
            # print(type(X_train))
            # print(len(X_train))
            # print(X_train.columns)
            
            # print(y_train)
            # print(type(y_train))
            # print(len(y_train))

            # print(X_test)
            # print(type(X_test))
            # print(len(X_test))
            

            # print(y_test)
            # print(type(y_test))
            # print(len(y_test))
            
            
            model.fit(X_train,y_train)
            
            
            # Make predictions on the test set
            predictions = model.predict(X_test) 
            
            # Evaluate model performance
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            print("Root Mean Squared Error:", rmse)

            ## INITIALIZE DF_PREDICTIONS DATAFRAME
            # For start create a new df containing the dates and the predictions
            df_predictions = pd.DataFrame(columns=['date', 'indicator_prediction_values'])

            # # Prediction: Use the trained model to predict future indicator values
            for i in range(0,len(df_ml)):
                if np.isnan(df_ml.iloc[i]['lag1']):
                    print('Nan value for row: {}'.format(i))
                else:
                    print('Found a value to make a prediction at row {}'.format(i))
                    latest_data = df_ml.iloc[i]['lag1']  # Get the most recent indicator value
                    print('Latest data tha value is not nan: {}'.format(latest_data))
                    ## Find the date from the df_plot that belongs to latest_data value in order to have the date
                    # print('Type of latest_data: {}'.format(type(latest_data))) # numpy.float64 doesn't matter tho

                    # Find the row from the df_plot that contains this indicator_value
                    # and get the date for this value. Save this value as the latest_date
                    print(df_plot[df_plot['indicator_values'] == latest_data]['date']) 
                    latest_date = df_plot[df_plot['indicator_values'] == latest_data]['date'].values[0]  # Get the most recent indicator value
                    print('Latest date: {} type: {}'.format(latest_date, type(latest_date)))
                    print('------------------------------------------------------------------------')
                    ## We should break the loop when we found our first indicator_value that is not nan
                    break
            if np.isnan(latest_data) == False:
                future_prediction = model.predict([[latest_data]])  # Predict future value based on latest data
                future_pred_round = round(future_prediction[0])
                print('Predicted Future Value: {} for country: {} Based on data'.format(future_prediction[0], country))

                ## The prediction date refers to next year from the year of the latest_date so we + 1 the year
                prediction_date = date(latest_date.year+1,latest_date.month, latest_date.day)  
                print('New date: {}'.format(prediction_date))

                # ## Create a new row that contains the prediction date and prediction value 
                # new_row = pd.DataFrame({'date':prediction_date, 'indicator_prediction_values': future_pred_round})

                # # Set the index for the new row DataFrame
                # new_row.set_index(pd.Index([0]), inplace=True)
                
                # # Concatenate the new row with the original DataFrame
                # df_predictions = pd.concat([df_predictions, new_row], ignore_index=True)

                # # Add the new row to the DataFrame at index 0
                # df_predictions.loc[0] = new_row

                # Create a new row as a dictionary
                new_row = {'date': '2023-01-01', 'indicator_prediction_values': 12345.67}


                num_predictions = 5  # Number of additional predictions
                for number_of_predictions in range(num_predictions):
                    future_prediction = model.predict([[latest_data]])  # Predict future value based on latest data
                    future_pred_round = round(future_prediction[0])
                    print('Predicted Future Value: {} Based on prediction. Step: {}'.format(future_pred_round,number_of_predictions+1))
                    latest_data = future_prediction[0]  # Update latest_data for the next prediction
                    # print(type(latest_data)) #numpy.float64
                    # print(type(future_prediction)) #numpy.ndarray
                    # print(type(future_prediction[0])) #numpy.float64

                    prediction_date = date(latest_date.year+1,latest_date.month, latest_date.day)  
                    print('New date: {}'.format(prediction_date))

                    ## Create a new row that contains the prediction date and prediction value 
                    # new_row = pd.DataFrame({'date':prediction_date, 'indicator_prediction_values': future_pred_round})

                    # # Set the index for the new row DataFrame
                    # new_row.set_index(pd.Index([0]), inplace=True)
                    
                    # df_predictions = pd.concat([df_predictions, new_row], ignore_index=True)


                    # Create a new row as a dictionary
                    new_row = {'date': '2023-01-01', 'indicator_prediction_values': 12345.67}

                    # Add the new row to the DataFrame at index 0
                    df_predictions.loc[0] = new_row
            else:
                print('All rows are Nan and the model cannot make prediction on this particular Indicator')
            
            ## Lets see what we got on our df_predictions
            print(df_predictions)
                




        
        fig = {
            'data': traces,
            'layout': go.Layout(title = 'Countries/Indicators for %s' %countries_list)
        }
        return fig

if __name__ == '__main__':
    app.run_server()