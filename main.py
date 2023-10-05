import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Setup der Streamlit-Page und laden des Datensatzes
st.set_page_config(layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv("autoscout24.csv")

####################################
# Kurzes Intro in den Datensatz

st.title('The Data Set')

st.write('Der Datensatz umfasst Neu- und Gebrauchtwagendaten der Website Autoscout24, Europas größtem Anbieter. Die gesammelten Daten reichen von 2011 bis 2021, wir werden anhand dieser probieren, die Preise verschiedener Automarken hervorzusagen.')

st.write(df.head())

####################################
# Darstellung der verschiedenen Marken
st.title('Welche Marken wurden erfasst?')

# Hier werden erst die Marken aus dem Dataframe in eine Liste exportiert und dann aus Layout-Gründen in einen einzelnen String zusammengefügt
unique_makes = [make for make in df['make'].unique()]
comma_separated_string = ', '.join(unique_makes)
st.write(comma_separated_string)


####################################
# Darstellung der Sales als Histogram nach Jahren
st.title('Car Sales Histogram')

# Hier werden erst die Daten des Dataframes nach Jahren gruppiert
sales_per_year = df.groupby('year').size().reset_index(name='count')

# Erstellung des Plots
fig, ax = plt.subplots(figsize=(8,3))
ax.bar(sales_per_year['year'], sales_per_year['count'])

# Limits der y-Achse, um Unterschiede zwischen den Jahren gut erkennen zu können
ax.set_ylim(4000, 4500)
# Ermittlung und Darstellung des MEAN der Jahre, um Referenzwert für Ausschlag der einzelnen Jahre zu haben.
average_sales = sales_per_year['count'].mean()
ax.axhline(y=average_sales, color='r', linestyle='--', label=f'Average ({average_sales:.0f})')


#ax.set_xlabel('Year')
ax.set_ylabel('Number of Sales')
ax.legend()
plt.xticks(range(2011, 2022))
plt.xticks(rotation=45)
st.pyplot(fig)

####################################
# Erstellung zweier Plots, die augenscheinlich interessant sind
st.title('Welche Korrelationen sind interessant?')

# Initialisierung zweier Spalten, um die Plots nebeneinander anzeigen zu können
left_column, right_column = st.columns(2)

# Horsepower Plot
with left_column:
    hp_fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.scatter(df['hp'], df['price'], alpha=0.5)
    ax1.set_xlabel('Horsepower (hp)')
    ax1.set_ylabel('Price (Million)')

    ax1.set_xlim(df['hp'].min(), df['hp'].max())

    st.pyplot(hp_fig)

# Mileage Plot
with right_column:
    mileage_fig, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(df['mileage'], df['price'], alpha=0.5)
    ax2.set_xlabel('Mileage (Million)')
    ax2.set_ylabel('Price (Million)')
    
    #print(f" Milage min: {df['mileage'].min()}")
    #print(f" Milage max: {df['mileage'].max()}")
    #print(f" Price min: {df['price'].min()}")
    #print(f" Price min: {df['price'].max()}")
    
    st.pyplot(mileage_fig)


####################################

st.title('Machine Learning')

#Erstellung eines Filter, der nur die fünf verkaufsstärksten Marken berücksichtigt
top_manufacturers = df['make'].value_counts().head(5)
top_manufacturers_list = top_manufacturers.index.tolist()


comma_separated_string = ', '.join(top_manufacturers_list)
st.write(f"Für das Machine Learning betrachten wir nur die fünf Marken, die die meistens Verkäufe ausmachen: {comma_separated_string}.")

top_df = df[df['make'].isin(top_manufacturers_list)]

#nan_count = filtered_df.isna().sum()
top_df.dropna(inplace=True)


# Ermittelung des durchschnittlichen Preises der Top 5
average_prices = top_df.groupby('make')['price'].mean()

######################################
# Machine Learning mit linearer Regression
st.subheader("Lineare Regression")

# Erstellung von X und y (Features und Target)
X = top_df[['mileage', 'hp', 'year']] 
y = top_df['price']

# Trennung in Trainings- und Testdatensatz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisierung des Models
model = LinearRegression()

# Fitten der Daten auf die Trainingsdaten
model.fit(X_train, y_train)

# Vorhersage der Preise anhand der Testdaten
y_pred = model.predict(X_test)
print(y_pred)

# Erstellung und Darstellen der Fehlermetriken
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write(f'Mean Absolute Error: {mae:.2f}') # Niedriger ist gut
st.write(f'Mean Squared Error: {mse:.2f}') # Niedriger ist gut
st.write(f'R-squared: {r2:.2f}') #Nah an 1 ist gut
#print(model.coef_)
#print(model.intercept_)

######################################
# Lineare Regression mit den einzelnen Features zusammen mit deren Plots

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Liste der Features für die For-Schleife
feature_names = ['Mileage', 'HP', 'Year']

for i, feature in enumerate(X.columns):
    # Fitten der Daten mit jeweiligem Feature und Target
    model = LinearRegression()
    model.fit(X_train[[feature]], y_train)

    # Prediction
    y_pred = model.predict(X_test[[feature]])

    # Erstellen der jeweiligen Plots für das Feature mit Regressiongerade
    ax[i].scatter(X_test[feature], y_test, color='blue', label='Actual Data', alpha=0.5)

    x_range = np.linspace(X_test[feature].min(), X_test[feature].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    ax[i].plot(x_range, y_range, color='red', linewidth=2, label='Linear Regression')
    
    ax[i].set_title(f'Linear Regression for {feature_names[i]}')
    ax[i].set_xlabel(feature_names[i])
    ax[i].set_ylabel('Price')
    ax[i].legend()
    ax[i].grid(True)

    # Anzeigen der verschiedenen Fehlermetriken in den jeweiligen Plots
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ax[i].text(0.1, 0.9, f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nR2: {r2:.2f}', transform=ax[i].transAxes, fontsize=12,
               verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})


st.pyplot(plt.tight_layout())

################################
# Lineare Regression mit den einzelnen Features zusammen mit deren Plots
st.subheader("K-Nearest Neighbors")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Auswahl von k = 5 (Elbow-Method), fitten der Daten
k = 5 
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)

# Prediction mit aufgeladenem Model
y_pred = model.predict(X_test)

# Erstellen der Fehlermetriken
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared: {r2:.2f}')

#############################

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

feature_names = ['Mileage', 'HP', 'Year']

for i, feature in enumerate(X.columns):
    # Fitten der Daten mit jeweiligem Feature und Target
    knn = KNeighborsRegressor(n_neighbors=5)  
    knn.fit(X_train[[feature]], y_train)

    # Prediction
    y_pred = knn.predict(X_test[[feature]])

    # Erstellen der jeweiligen Plots für das Feature mit Regressiongerade
    ax[i].scatter(X_test[feature], y_test, color='blue', label='Actual Data', alpha=0.5)

    x_range = np.linspace(X_test[feature].min(), X_test[feature].max(), 100)
    y_range = knn.predict(x_range.reshape(-1, 1))
    ax[i].plot(x_range, y_range, color='green', linewidth=2, label='KNN Regression')
    
    ax[i].set_title(f'KNN Regression for {feature_names[i]}')
    ax[i].set_xlabel(feature_names[i])
    ax[i].set_ylabel('Price')
    ax[i].legend()
    ax[i].grid(True)

     # Anzeigen der verschiedenen Fehlermetriken in den jeweiligen Plots
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ax[i].text(0.1, 0.9, f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nR2: {r2:.2f}', transform=ax[i].transAxes, fontsize=12,
               verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})

st.pyplot(plt.tight_layout())

st.title("Fazit")
st.write("Einzelne Features performen bei KNN besser, alle drei aber bei Lineare Regression besser.")

##################################

st.title("Ansatz mit kategorischen Features")
st.write("Die Ergebnisse mit rein numerischen Features sind sicher verbesserungswürdig, da natürlich Marke oder Kraftstoff des Autos maßgeblich den Preis beeinflussen.")

# Bearbeiten der Daten nach intensiverer Datenanalyse
# Null-Values entfernen, Horsepower entfernen (verzerrt Ergebisse aufgrund von Inseraten wie Wohnwägen), Kraftstoffe auf gängige Kraftstoffe begrenzen, Firmenwägen ausschließen sowie extrem hochpreisige Autos, um möglichst repräsentative Werte zu haben
df = df.dropna()
df=df.drop(['hp'], axis=1)
df=df[df['fuel']!="-/- (Fuel)"]
df=df[df['fuel']!="Others"]
df=df[df['offerType']!="Employee's car"]
df=df[df['price']<500000]
df=df.reset_index(drop=True)

# Für die Prediction-App in der zweiten Streamlit-Page die Daten in den Session-State (Cache) übergeben
if 'final_df' not in st.session_state:
    st.session_state.final_df = df

###################################
# Erneutes Erstellen von Trainings- und Testdaten
X = df.drop(columns='price')
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Aufladen des Models mit einen Teil der Daten (den kategorischen Features) zur Umwandlung dieser in numerische Werte
ohe = OneHotEncoder()
ohe.fit(X[['make', 'model', 'fuel', 'gear', 'offerType']])

# Spaltentransformer, der die transformierten Daten weitergibt und kommuniziert, dass die restlichen Daten (numerische Werte) untransformiert weitergegeben werden
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['make', 'model', 'fuel', 'gear', 'offerType']), 
                                      remainder='passthrough')

# Erstellung des Modells und einer Pipeline, die transformierten und untransformieten Daten und erstelles Modell nutzt
rf = RandomForestRegressor()
pipe = make_pipeline(column_trans, rf)

# Fitten der Pipeline mit den Daten 
pipe.fit(X_train, y_train)

# Prediction
y_pred_train_rf = pipe.predict(X_train)

# Erstellen der Fehlermetriken
r2 = r2_score(y_train, y_pred_train_rf)
mae = mean_absolute_error(y_train, y_pred_train_rf)
mse = mean_squared_error(y_train, y_pred_train_rf)

st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared: {r2:.2f}')

##############################
# Vergleich einer Auswahl der Preise in Tabellenform
comparerf = pd.DataFrame()
comparerf['Actual Price'] = y_train
comparerf['Predicted Price'] = y_pred_train_rf.round(2)
st.write(comparerf.head(15))

###############################
# Vergleich einer Auswahl der Preise in Plotform
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(comparerf.index[:15], comparerf['Actual Price'][:15], label='Actual Price', color='blue')
ax.scatter(comparerf.index[:15], comparerf['Predicted Price'][:15], label='Predicted Price', color='red')

ax.set_xlabel('Index')
ax.set_ylabel('Price')
ax.set_title('Echter Preis vs. Predicted Preis')
ax.legend()

st.pyplot(plt.tight_layout())

####################################

# Exportieren der Pipe für die zweite Streamlitpage, sodass sie einfacher zu nutzen ist
joblib.dump(pipe, 'trained_pipeline.pkl')


