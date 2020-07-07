# Importing librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# ----------------------------------------------------------------------------------------------
def compare_plot(l_series, title):
    
    plt.figure(figsize = (15,10))
    plt.suptitle(title, fontweight = 'heavy')

    plt.subplot(221)
    l_series[0].plot(kind = 'box')
    plt.grid(True)
    plt.title('2015 - All values', fontweight = 'bold')
    plt.ylabel("Values")

    plt.subplot(222)
    l_series[1].plot(kind = 'box')
    plt.grid(True)
    plt.title('2016 - All values', fontweight = 'bold')

    plt.subplot(223)
    l_series[0].plot(kind = 'box', showfliers = False)
    plt.grid(True)
    plt.title('2015 - Without outliers', fontweight = 'bold')
    plt.ylabel("Values")

    plt.subplot(224)
    l_series[1].plot(kind = 'box', showfliers = False)
    plt.grid(True)
    plt.title('2016 - Without outliers', fontweight = 'bold')

    plt.show()

# ----------------------------------------------------------------------------------------------
def df_fillrates(df, col = 'selected columns', h_size = 15):
    """ Returns a barplot showing for each column of a dataset df the percent of non-null values """

    nb_columns = len(df.columns)
    df_fillrate = pd.DataFrame(df.count()/df.shape[0])
    df_fillrate.plot.barh(figsize = (h_size, nb_columns/2))
    
    plt.title("Fillrate of columns in {}".format(col), fontweight = 'bold', fontsize = 12)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.gca().legend().set_visible(False)
    plt.show()
    
# ----------------------------------------------------------------------------------------------
def score_predictions(y_true, y_predict):
    
    from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
    
    # Calculation of scores
    r2_gas = round(r2_score(y_true.iloc[:,0], y_predict.iloc[:,0]),4)
    r2_energy = round(r2_score(y_true.iloc[:,1], y_predict.iloc[:,1]),4)
    
    try:
        msle_gas = round(mean_squared_log_error(y_true.iloc[:,0], y_predict.iloc[:,0]),4)
    except:
        msle_gas = np.NaN
    
    try:
        msle_energy = round(mean_squared_log_error(y_true.iloc[:,1], y_predict.iloc[:,1]),4)
    except:
        msle_energy = np.NaN
    
    # Displaying scores
    
    df_scores = pd.DataFrame(
        np.array([
            [msle_gas, msle_energy], [r2_gas, r2_energy]
        ]),
        columns = ['GHG emissions', 'Energy use'],
        index = ['MSLE', 'R²']
    )
    
    return df_scores  

# ----------------------------------------------------------------------------------------------
def store_results(y, model_name, df_storage):
    """ Stores model predictions in passed dataframe"""
    
    if type(y) == np.ndarray:
        df_storage[model_name + '_GHG'] = y[:,0]
        df_storage[model_name + '_Energy'] = y[:,1]
    else :
        df_storage[model_name + '_GHG'] = y.iloc[:,0].values
        df_storage[model_name + '_Energy'] = y.iloc[:,1].values
    
    return df_storage

# ----------------------------------------------------------------------------------------------
def plot_results(df, model_name, title, p_score,
                 scale_ghg_x = 'symlog', scale_ghg_y = 'symlog', 
                 scale_eng_x = 'symlog', scale_eng_y = 'symlog'):

    # Getting scores
    df_scores = score_predictions(df[['test_GHG', 'test_Energy']], 
                                  df[[model_name + '_GHG', model_name + '_Energy']])
    
    score_gas = df_scores.loc[p_score, 'GHG emissions']
    score_eng = df_scores.loc[p_score, 'Energy use']
    
    plt.figure(figsize = (20,10))
    
    # Plotting GHG Emissions
    plt.subplot(121)
    plt.scatter(df['test_GHG'], df[model_name + '_GHG'])
    plt.xlabel('True value', backgroundcolor = 'lightgrey', fontstyle = 'italic')
    plt.ylabel('Predicted value', backgroundcolor = 'lightgrey', fontstyle = 'italic')
    
    plt.plot([df['test_GHG'].min(), df['test_GHG'].max()],
             [df['test_GHG'].min(), df['test_GHG'].max()], color = 'grey')
        
    plt.gca().set_xscale(scale_ghg_x)
    plt.gca().set_yscale(scale_ghg_y)
    
    plt.grid(True)
    plt.title('Greenhouse Gas Emissions - {} = {}'.format(p_score, score_gas), 
              fontweight = 'bold')

    # Plotting Energy use
    plt.subplot(122)
    plt.scatter(df['test_Energy'], df[model_name + '_Energy'])
    plt.xlabel('True value', backgroundcolor = 'lightgrey', fontstyle = 'italic')
    plt.ylabel('Predicted value', backgroundcolor = 'lightgrey', fontstyle = 'italic')
    
    plt.plot([df['test_Energy'].min(), df['test_Energy'].max()], 
             [df['test_Energy'].min(), df['test_Energy'].max()], color = 'grey')
    
    plt.gca().set_xscale(scale_eng_x)
    plt.gca().set_yscale(scale_eng_y)
    
    plt.grid(True)
    plt.title('Total Energy Use - {} = {}'.format(p_score, score_eng), fontweight = 'bold')

    plt.suptitle(title, fontweight = 'bold', fontsize = 15)
    
plt.show()

