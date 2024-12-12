# 🔧 Basic functions
def snake_columns(data): 
    """
    returns the columns in snake case
    """
    data.columns = [column.lower().replace(' ', '_') for column in data.columns]
    
def open_data(data): # returns shape, data types & shows a small sample
    print(f"Data shape is {data.shape}.")
    print()
    print(data.dtypes)
    print()
    print("Data row sample and full columns:")
    return data.sample(5)

# 🎯 Specific functions
def explore_data(data): # sum & returns duplicates, NaN & empty spaces
    duplicate_rows = data.duplicated().sum()
    nan_values = data.isna().sum()
    empty_spaces = data.eq(' ').sum()
    nan_pct = round((data.isna().sum()/len(data)*100),2)
    import pandas as pd
    exploration = pd.DataFrame({"NaN": nan_values, "NaN %": nan_pct, "EmptySpaces": empty_spaces}) # New dataframe with the results
    print(f"There are {duplicate_rows} duplicate rows. Also;")
    return exploration

def outliers(data, threshold = 1.5, result = None): 
    """
    Identifies the outliers in each column and returns a dataframe with only the outliers.

    If the parameter result = "slayed" it returns the dataframe passed in without outliers
    """
    import numpy as np
    import pandas as pd
    outliers=[]
    for column in data.select_dtypes(include=[np.number]):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        if result == "slayed":
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        else:
            outliers.append(data[(data[column] < lower_bound) | (data[column] > upper_bound)])
    if result == "slayed":
        return data
    else:
        outliers_df = pd.concat(outliers).drop_duplicates()
        return outliers_df


def evaluate_regresion_model(df, target, size=0.3, random=42, shuff=True, round_digits=4, graphs=False, stand=False, log=False):
    """
    Function to automatically evaluate ML regresion models and returns a df with the comparison
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    # Definition of the models
    models = {
        'Linear Regression': LinearRegression(),  # Simple linear regression
        'Ridge Regression': Ridge(),  # Regularized regression (L2)
        'Lasso Regression': Lasso(),  # Regularized regression (L1)
        'Elastic Net Regression': ElasticNet(),  # Combination of Ridge and Lasso
        'Polynomial Regression (degree=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),  # Polynomial regression
        'Decision Tree Regression': DecisionTreeRegressor(),  # Decision tree
        'Random Forest Regression': RandomForestRegressor(),  # Random forest
        'K-Nearest Neighbors Regression': KNeighborsRegressor(),  # KNN
        'Support Vector Regression': SVR(),  # Support vector machine for regression
        'Neural Network Regression': MLPRegressor(max_iter=500),  # Multi-layer perceptron (neural network)
        'Bayesian Regression': BayesianRidge(),  # Bayesian regression
        'Gradient Boosting Regression': GradientBoostingRegressor(),  # Gradient boosting
        'XGBoost Regression': xgb.XGBRegressor(),  # XGBoost boosting
        'XGBRF Regression': xgb.XGBRFRegressor()  # XGBoost Random Forest
        }

    metrics = {        
        'Model': [],
        'R²': [],
        'RMSE': [],
        'MSE': [],
        'MAE': []
        }
    
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Separate data into train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = random, shuffle = shuff)

    # Normalization with StandardScaler:
    if stand == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("----------------------Data Normalized with Standard Scaler-------------------------")

    # Normalization with Log Transform:
    if log == True:
        X_train = np.log1p(X_train)  
        X_test = np.log1p(X_test)
        print("----------------------Data Normalized with Log Transform-------------------------")

    #Run all the models and store the results
    predictions_dict = {}
    for key, model in models.items():
        #print(key)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics['Model'].append(key)
        metrics['R²'].append(r2_score(y_test, predictions))
        metrics['RMSE'].append(root_mean_squared_error(y_test, predictions))
        metrics['MSE'].append(mean_squared_error(y_test, predictions))
        metrics['MAE'].append(mean_absolute_error(y_test, predictions))
        predictions_dict[key] = predictions

    # Convert metrics to DataFrame
    df_metrics = round(pd.DataFrame(metrics), round_digits).set_index("Model").T

    # Plot subplots for Actual vs. Predicted if graphs=True
    if graphs:
        n_models = len(models)
        nrows = math.ceil(n_models / 3)  # Adjust row count to have 3 plots per row
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(20, 5 * nrows))
        axes = axes.flatten()

        for i, (key, predictions) in enumerate(predictions_dict.items()):
            comparison_df = pd.DataFrame({'actual': y_test, 'predictions': predictions}).reset_index(drop=True)
            sns.regplot(x='actual', y='predictions', data=comparison_df, scatter_kws={"color": "#FF6347", "alpha": 0.7},
                        line_kws={"color": "#FF8C00", "linewidth": 3}, ax=axes[i])
            axes[i].set_title(f'{key}')
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predictions')

        # Hide any empty subplots if n_models is not a multiple of 3
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    
    return df_metrics

def range_verifier(value):
    return isinstance(value, range)

def evaluate_clasification_model(df, target, size=0.3, random=42, shuff=True, neighbors=3, trees=3):
    """
    Function to automatically evaluate ML clasification models and shows results.
    Parameters neighbors and tree can be:
     - integers to define the number of neighbors and depth of tree levels
     - ranges (ex. range(5, 30)) to tune the parameter
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression 
    import xgboost as xgb
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report
    import matplotlib.pyplot as plt
    import numpy as np

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Take only the numerical features
    X = X.select_dtypes(include=[np.number])

    # Separate data into train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = random, shuffle = shuff)

    # Determine the optimal number of neighbors if `neighbors` is a range
    if range_verifier(neighbors) == False:
        num_neighbors = neighbors
    else:
        n_test = {}
        for neighbor in neighbors:
            model = KNeighborsClassifier(n_neighbors=neighbor)
            model.fit(X_train, y_train)
            n_test[neighbor] = model.score(X_test,y_test)
        num_neighbors = max(n_test, key=n_test.get)

    # Determine the optimal tree depth if `trees` is a range
    if range_verifier(trees) == False:
        num_trees = trees
    else:
        t_test = {}
        for tree in trees:
            model = DecisionTreeClassifier(max_depth=tree)
            model.fit(X_train, y_train)
            t_test[tree] = model.score(X_test,y_test)
        num_trees = max(t_test, key=t_test.get)

    # Definition of the models
    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(n_neighbors=num_neighbors),
        'Decision Tree': DecisionTreeClassifier(max_depth=num_trees),
        'SVM': LinearSVC(),
        'XGBoost': xgb.XGBClassifier()
        }
    
    #Run all the models and store the results
    for key, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("\n---------------------------------------------------\n")
        print(key) # Print the model name
        if key == "KNN":
            print(f"with {num_neighbors} neighbors")
        elif key == "Decision Tree":
            print(f"with {num_trees} maximum depth")
        
        print(classification_report(y_test, predictions)) # Print the model clasification report
        
        # Print the acurracy of test and train
        print("Test data accuracy: ",model.score(X_test,y_test))
        print("Train data accuracy: ", model.score(X_train, y_train))
        
        # Calculate and displays the confusion matrix
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure(figsize=(8, 6))
        disp.plot(cmap='Blues')
        if key == "KNN":
            plt.title(f'{key} Confusion Matrix with {num_neighbors} neighbors')
        elif key == "Decision Tree":
            plt.title(f'{key} Confusion Matrix with {num_trees} maximum depth')
        else:
            plt.title(f'{key} Confusion Matrix')
        plt.grid(False)
        plt.show()

        # Calculate and displays the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, predictions, pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{key} Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(False)
        plt.show()

def correlation_matrix(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = data.select_dtypes(include=[np.number])
    num_corr = round(data.corr(), 2)
    # Correlation Matrix-Heatmap Plot
    mask = np.zeros_like(num_corr)
    mask[np.triu_indices_from(mask)] = True # optional, to hide repeat half of the matrix

    f, ax = plt.subplots(figsize=(25, 15))
    sns.set(font_scale=1.5) # increase font size

    ax = sns.heatmap(num_corr, mask=mask, annot=True, annot_kws={"size": 12}, linewidths=.5, cmap="coolwarm", fmt=".2f", ax=ax) # round to 2 decimal places
    ax.set_title("Dealing with Multicollinearity", fontsize=20) # add title
    plt.show()


def distribution_check(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    color = '#13599B'
    # grid size
    nrows, ncols = 5, 4  # adjust for your number of features
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    axes = axes.flatten()
    # Plot each numerical feature
    for i, ax in enumerate(axes):
        if i >= len(data.columns):
            ax.set_visible(False)  # hide unesed plots
            continue
        ax.hist(data.iloc[:, i], bins=30, color=color, edgecolor='black')
        ax.set_title(data.columns[i])
    plt.tight_layout()
    plt.show()


def check_outliers(data, number = 1.5):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    data = data.select_dtypes(include=[np.number])
    color = '#13599B'
    # grid size
    nrows, ncols = 5, 4 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(data.columns):
            ax.set_visible(False)
            continue
        ax.boxplot(data.iloc[:, i].dropna(), vert=False, patch_artist=True,
                whis=number, # Ajusta este valor para controlar la longitud de los bigotes 
                boxprops=dict(facecolor=color, color='black'), 
                medianprops=dict(color='yellow'), whiskerprops=dict(color='black'), 
                capprops=dict(color='black'), flierprops=dict(marker='o', color='red', markersize=5))
        ax.set_title(data.columns[i], fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    plt.show()