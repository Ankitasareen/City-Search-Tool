import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.title("City Choosing App")

#@st.cache(persist = True)
def load_movehuncostofliving(nrows):
    data1 = pd.read_csv("movehubcostofliving.csv", nrows=nrows)
    return data1

#@st.cache(persist = True)
def load_lat(nrows):
    data2 = pd.read_csv("lat.csv" , nrows = nrows)
    return data2


df_qual = load_lat(217)
df_cost = load_movehuncostofliving(217)

df_qual.sort_values('City',inplace=True,ignore_index=True)
df_cost.sort_values('City',inplace=True,ignore_index=True)

df = pd.concat([df_cost,df_qual.drop(['City'],axis=1)],axis=1)
df.columns = df.columns.str.replace(' ','_')

#st.subheader("Data")
#st.write(df)

le=LabelEncoder()
df_no_city =df.drop(['City'],axis=1)
df_city = df.iloc[:,0:1]
le_city = le.fit_transform(df_city)
df_city = pd.DataFrame(le_city,columns=['LE'])
df_k_means = pd.concat([df_city,df],axis = 1)

X1=df_k_means.iloc[:,4:5].values

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    valueScaled = float(value - leftMin) / float(leftSpan)

    return rightMin + (valueScaled * rightSpan)

df['Overall_Score']=0
inc_features = [ 'Movehub_Rating', 'Purchase_Power', 'Health_Care', 
       'Quality_of_Life', 'Avg_Disposable_Income']
dec_features = ['Pollution','Crime_Rating','Cappuccino', 'Cinema',
       'Wine', 'Gasoline', 'Avg_Rent']
for i in range(0,216):
  score = []
  for feat in inc_features:
    val = df[feat][i]
    var = translate(val,df[feat].min(),df[feat].max(),0,10)
    score.append(var)
  for feat in dec_features:
    val = df[feat][i]
    var = translate(val,df[feat].min(),df[feat].max(),10,0)
    score.append(var)
  df['Overall_Score'][i]= sum(score)/12
 
 


features=['City', 'Movehub_Rating', 'Purchase_Power', 'Health_Care', 'Pollution',
       'Quality_of_Life', 'Crime_Rating', 'Cappuccino', 'Cinema',
       'Wine', 'Gasoline', 'Avg_Rent', 'Avg_Disposable_Income',
       'Overall_Score']

st.title("CITY MAP ACCORDING TO OVERALL RATING")
st.write("Here you can see the average score of all the cities across on the basis of different parameters such as crime rate, pollution,rent etc.")
fig = px.scatter_mapbox(df,
                        lat="lat", lon="lon", color="Overall_Score", hover_name="City",
                        hover_data=features,
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=1,
                        mapbox_style="carto-positron")
st.plotly_chart(fig)
  
st.title("CHOOSE YOUR HOMETOWN WISELY...")
st.write("Here , you can see the score of all the cities on the basis of a particular parameter of your choice.")
option = st.selectbox('What features do you prefer?' , 
                        ('City', 'Movehub_Rating', 'Purchase_Power', 'Health_Care', 'Pollution',
       'Quality_of_Life', 'Crime_Rating', 'Cappuccino', 'Cinema',
       'Wine', 'Gasoline', 'Avg_Rent', 'Avg_Disposable_Income',
       'Overall_Score')
       
  )


st.write("You selected:" , option)
  

fig = px.scatter_mapbox(df,
                        lat="lat", lon="lon", color=option, hover_name="City",
                        hover_data=['City',option],
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=1,
                        mapbox_style="carto-positron")
st.plotly_chart(fig)

##########################################################################################
st.title("Find out which features matter the most in your city recommendation")
st.write("Here, in this piechart by selecting a city we can see how much a factor is important for that city.")
user_input = st.text_input("Enter City" )

City = user_input
inc_features = [ 'Movehub_Rating', 'Purchase_Power', 'Health_Care', 
       'Quality_of_Life', 'Avg_Disposable_Income']
dec_features = ['Pollution','Crime_Rating','Cappuccino', 'Cinema',
       'Wine', 'Gasoline', 'Avg_Rent']
score = []

df_pie = df[df.City == City]
for feat in inc_features:
  val = df_pie[feat]
  var = translate(val,df[feat].min(),df[feat].max(),0,10)
  score.append(var)
for feat in dec_features:
  val = df_pie[feat]
  var = translate(val,df[feat].min(),df[feat].max(),10,0)
  score.append(var)

labels = 'Movehub_Rating', 'Purchase_Power', 'Health_Care','Quality_of_Life', 'Avg_Disposable_Income','Pollution','Crime_Rating','Cappuccino', 'Cinema','Wine', 'Gasoline', 'Avg_Rent'

fig1, ax1 = plt.subplots()
ax1.pie(score,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(City)
st.pyplot(fig1)

#################################################################################

st.title("Customised City Choosing tool")
st.write("Here, in this slider you can choose any values")

Movehub_Rating = st.slider("Movehub Rating", min_value=0, max_value=100, value=70, step=None, format=None, key=None)
Purchase_Power = st.slider("Purchase Power", min_value=0, max_value=100, value=1, step=None, format=None, key=None)
Health_Care = st.slider("Health_Care", min_value=0, max_value=100, value=80, step=None, format=None, key=None)
Quality_of_Life = st.slider("Quality_of_Life", min_value=0, max_value=100, value=0, step=None, format=None, key=None)
Avg_Disposable_Income =st.slider('Avg_Disposable_Income', min_value=0, max_value=5000, value=0, step=None, format=None, key=None)
Pollution = st.slider('Pollution', min_value=0, max_value=100, value=100, step=None, format=None, key=None)
Crime_Rating = st.slider('Crime_Rating', min_value=0, max_value=100, value=100, step=None, format=None, key=None)
Cappuccino = st.slider('Cappuccino', min_value=0, max_value=5, value=5, step=None, format=None, key=None)
Cinema = st.slider('Cinema', min_value=0, max_value=100 , value=100, step=None, format=None, key=None)
Wine = st.slider('Wine', min_value=0, max_value=30, value=30, step=None, format=None, key=None)
Gasoline = st.slider('Gasoline', min_value=0, max_value=2, value=2, step=None, format=None, key=None)
Avg_Rent = st.slider( 'Avg_Rent', min_value=0, max_value=6000, value=600, step=None, format=None, key=None)

df1 = df[((df.Movehub_Rating )>float(Movehub_Rating))&((df.Purchase_Power)>float(Purchase_Power))&
         ((df.Health_Care )>float(Health_Care))& ((df.Pollution )<float(Pollution))
         &((df.Crime_Rating )<float(Crime_Rating))&((df.Avg_Rent )<float(Avg_Rent))&((df.Avg_Disposable_Income )>float(Avg_Disposable_Income))
         &((df.Gasoline )<float(Gasoline))&((df.Cinema )<int(Cinema))&((df.Wine )<int(Wine))&((df.Cappuccino )<int(Cappuccino))]

features = df.columns
fig5 = px.scatter_mapbox(df1,
                        lat="lat", lon="lon",  hover_name="City",
                        hover_data=features,color='Overall_Score',
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=1,
                        mapbox_style="carto-positron")
st.plotly_chart(fig5)
df1.sort_values('Overall_Score',ascending=False)


#########################################################################################3

st.title("GROUPED ACCORDING TO WHAT FEATURES YOU CHOOSE")
st.write("Here we are clustering the cities in groups according to various choices made by the user")
 
le=LabelEncoder()
df_no_city =df.drop(['City'],axis=1)
df_city = df.iloc[:,0:1]
le_city = le.fit_transform(df_city)
df_city = pd.DataFrame(le_city,columns=['LE'])
df_k_means = pd.concat([df_city,df],axis = 1)


st.write("GROUPED ACCORDING TO WHAT FEATURES YOU CHOOSE")
option2 = st.selectbox('What features do you prefer?' , 
                        ('City', 'Movehub_Rating', 'Purchase_Power', 'Health_Care', 'Pollution',
       'Quality_of_Life', 'Crime_Rating', 'Cappuccino', 'Cinema',
       'Wine', 'Gasoline', 'Avg_Rent', 'Avg_Disposable_Income',
       'Overall_Score') , key = "1"
       
  )


st.write("You selected:" , option2)

X1=df_k_means[[option2]]
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)


kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X1)
X1=np.squeeze(X1)

features = ['Movehub_Rating', 'Purchase_Power', 'Health_Care', 'Quality_of_Life', 'Pollution', 'Crime_Rating']


df['Cluster']=pd.Series(y_kmeans)
fig2 = px.scatter_mapbox(df,
                        lat="lat", lon="lon", color="Cluster", hover_name="City",
                        hover_data=features,
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=1,
                        mapbox_style="carto-positron")

st.plotly_chart(fig2)
#######################################################################################################################
