import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import community as co #Usamos esta libreria para las comunidades SNA
from libcommon import *
from libgraph import *

filter = "square"
filter_square = [64,90,74,78,73,107,79]
filter_district = filter_square + [78, 84, 77, 79, 76, 82, 73, 72, 63, 62, 61]
if filter=="square":
  retiro_stations = filter_square
  file = '201808_Usage_Bicimad.json.square'
else:
  retiro_stations = filter_district
  file = '201808_Usage_Bicimad.json.district'


"""#Carga de Dataset
A partir del módulo 3 y 4, se disponía de contenido tanto en HDFS como en Hive de todos los registros relacionados con los datos de Uso de las Bicicletas.

Por problemas de espacio, ya se ha hecho una labor previa de filtrado con las estaciones pertenecientes al retiro.

Se contemplan dos escenarios:


*   Estaciones que está alrededor del Parque Retiro (201808_Usage_Bicimad.json.square.filter).
*   Estaciones que pertenecen al Distrito Retiro (201808_Usage_Bicimad.json.district.filter)

Incluiremos, además un campo **weekend** que identifique si el día corresponde a Día Laboral (0) o Fin de Semana (1).

Declararemos dataFrames auxiliares para disponer de descripciones de ciertas variables categóricas: **user_type, ageRange, idplug_station, idunplug_station**.
La información de las estaciones se ha capturado usando la API BiciMad (módulo 3 y 4) y generando un **station.csv** para su carga. Ese fichero contiene el id de la estación, nombre, total_bases así como latitud y longitud.
"""

# Relación de variables categóricas user_type y ageRange
dfUserType = pd.DataFrame({
    'id_user_type': [0, 1, 2, 3], 
    'desc_user_type': ['Unknown', 'Annual', 'Occasional', 'Company']
})
dfAgeRange = pd.DataFrame({
    'id_ageRange': [0, 1, 2, 3, 4, 5, 6], 
    'desc_ageRange': [
        'Unknown', 
        '00-16 years', 
        '17-18 years', 
        '19-26 years', 
        '27-40 years', 
        '41-65 years', 
        '66- years']
})
# Relación de id de Estaciones 
dfStations = pd.read_csv('stations.csv', sep=';')
dfStations.head(5)

dfTrips = pd.read_csv(file + '.filter', sep=';', dtype={
    'datesample':'str', 
    'oid':'str', 
    'user_day_code':'str', 
    'idplug_base':'int64', 
    'user_type':'int64', 
    'idunplug_base':'int64', 
    'travel_time':'int64', 
    'idunplug_station':'int64', 
    'ageRange':'int64', 
    'idplug_station':'int64', 
    'zip_code':'str', 
    'longitude':'float64', 
    'latitude':'float64', 
    'var':'str', 
    'speed':'float64', 
    'secondsfromstart':'float64', 
    'daysample':'str'
})


'''
Filtrado de los viajes de las estaciones de retiro

dfTrips = dfTrips.loc[
  (dfTrips['idunplug_station'].isin(retiro_stations)) & 
  (dfTrips['idunplug_station'].isin(filter_square)
)].reset_index()
'''

#dfSpeeds = calculateSpeed(dfTrips)
#Usamos fichero de Speeds precalculado para optimizar tiempo de procesado.
dfSpeeds = pd.read_csv(file + '.speed', sep=';')
dfSpeeds['travel_space_avg'] = dfSpeeds['avg_speed']*dfSpeeds['travel_time_track']
dfSpeeds['travel_space_track'] = dfSpeeds['avg_track_speed']*dfSpeeds['travel_time_track']
#Eliminamos los Tracks y nos quedamos solo con informacion de viaje
dfTrips.drop([
		'idplug_base', 
		'idunplug_base', 
		'zip_code', 
		'longitude', 
		'latitude', 
		'var', 
		'speed', 
		'secondsfromstart'
	], axis=1, inplace=True)
dfTrips.drop_duplicates(inplace=True)

#Añadimos información de velocidades
dfTrips = dfTrips.merge(dfSpeeds, left_on='oid', right_on='oid', how='inner')

#Añadimos descripcion de user_type
dfTrips = dfTrips.merge(dfUserType, 
  left_on=['user_type'], right_on=['id_user_type'],
  how='inner'
)

#Añadimos descripcion de ageRange
dfTrips = dfTrips.merge(dfAgeRange, 
  left_on=['ageRange'], right_on=['id_ageRange'],
  how='inner'
)

#Añadimos informacion de estación de enganche
dfTrips = dfTrips.merge(dfStations, 
  left_on=['idplug_station'], right_on=['id'],
  how='inner'
)
dfTrips.rename(columns={
    'latitude': 'plug_station_latitude', 
    'longitude': 'plug_station_longitude',
    'name': 'plug_station_name',
    'number': 'plug_station_number',
    'address': 'plug_station_address',
    'total_bases': 'plug_station_total_bases'
}, inplace=True)

#Añadimos informacion de estación de des enganche
dfTrips = dfTrips.merge(dfStations, 
  left_on=['idunplug_station'], right_on=['id'],
  how='inner'
)
dfTrips.rename(columns={
    'latitude': 'unplug_station_latitude', 
    'longitude': 'unplug_station_longitude',
    'name': 'unplug_station_name',
    'number': 'unplug_station_number',
    'address': 'unplug_station_address',
    'total_bases': 'unplug_station_total_bases'
}, inplace=True)
dfTrips.drop(['ageRange', 'user_type', 'id_x', 'id_y'], axis=1, inplace=True)

dfTrips = addFeatures(dfTrips, retiro_stations)

dfTrips.head()

"""##Usuarios
Parece Interesante agrupar todos los viajes del dataset para hacer un estudio desde el punto de vista de usuarios, por ejemplo, cuántos viajes realizan o tiempo medio de los viajes. Destacar que la información que se puede agregar de usuarios es por día; variable user_day_code.
"""

user_vars = [
    'user_day_code', 
    'id_user_type', 
    'desc_user_type',
    'id_ageRange',
    'desc_ageRange',
    'day_of_week',
    'weekend'
]
dfUsers = dfTrips[['oid','travel_time'] + user_vars].drop_duplicates()

dfUsers = dfUsers.groupby(user_vars).agg({
					'oid':[('total_trips', 'count')],
					'travel_time':[('avg_travel_time', 'mean')]
}).reset_index()
dfUsers.columns = [b if b!="" else a for a, b in dfUsers.columns]

dfUsers.head()

"""#Análisis Exploratorio
Utilizaremos las funciones propias de Pandas info y describe para hacer un primer análisis. Destacar que aunque las variables son numéricas, realmente, muchas de ellas deberían considerarse como categóricas;por ejemplo: **user_type,  ageRange, idplug_station e idunplug_station**.

##Usuarios

A continuación se va a analizar el dataset haciendo foco en el usuario, para ver si tiene sentido estudiar el comportamiento teniendo como centro o no. 
En resumen que se ve abajo, es posible ya vislumbrar que la gran mayoría de tipos de usuario son Anuales, como indica que en el tercer cuartil predomina el tipo 1. Además otra pista que puede llegar a dar este resumen es que la mayoria de usuarios hacen un único viaje al día, como lo indica el tercer cuartil. Incluso en la media de tiempo por viaje del usuario se puede ver la diferencia entre el tercer cuartil y el maximo, siendo indicador de posibles outliers.
"""

dfUsers.describe()

dfUsers.info()

"""### Distribución total_trips y user_type"""

filter_var = {'name':'desc_user_type', 'desc': 'User Type'}
x_var = 'total_trips'
cols = 3

generateHistogram(filter_var, x_var, dfUsers, cols)

"""Con el visual de los histogramas de frecuencia por tipo de usuario, queda claro lo que se empezaba a entender en el resumen, la gran mayoria de usuarios hacen un único viaje. Empieza a entenderse por los puntos más altós en cada tipo de usuario que, según su modelo de contratación del servicio, predomina puede ser el Anual por encima del ocasional y el de empresa.

Detalles interesantes:
- Tipo Anual y Ocasional hay 1,2,3 viajes y su mayoría predomina 1 viaje (10700 users)
- Empresa hay hasta 14 grupos de viajes predomina 1 viaje, y rango de 2 a 10 viajes por usuario

### Distribución por total_trips y ageRange
"""

filter_var = {'name':'desc_ageRange', 'desc': 'Age Range'}
x_var = 'total_trips'
cols = 2
  
generateHistogram(filter_var, x_var, dfUsers, cols)

"""Si se cambia el tipo de usuario por el rango de edad ocurre exactamente igual, no hay rango de edad en la que no predomine de manera importante un solo viaje por día. De este modo puede a confirmarse que mayoritariamente en el dataset de agosto de 2018 en la zona del Parque del Retiro los usuarios que hicieron uso del servicio de Bicimad fue un único viaje por día. 

Datos de interes:
- Hay 7 tramos de edad (Unknown, <16,17-18,19-26,27-40,41-65,>66)
- Los tramos con más volumen después del Unknown es de 27-40, seguido de 41-65

A continuación vamos a ver de modo más visual el comportamiento de los tipos de usuario y rango de edad. Ya que en el primero se empezaba a ver un predomino importante de uno de los tipos de abono.

### Pie ageRange por weekend
"""

x_var = 'desc_ageRange'
filter_var = 'weekend'
generatePie(x_var, filter_var, dfUsers)

"""Con los siguientes gráficos de tartas puede entenderse como un gran desconocimiento del dato de rango de edad del dataset, 45% de Unknown, que puede llegar a afectar a la hora de clusterizar. Además se ve como grupos que mayor uso hacen de Bicimad los que estan entre 27-40 como principales y secundarios 41-65. El resto de rangos se ve un  uso menor del servicio. 

Datos de interes:
- El uso es mayor en laboral que fin de semana para todos los tramos de edad.
- La distribución de tramos de edad y uso según el tipo de día es equivalente aparentemente en proporción tanto para laboral como finde, la tendencia es que el uso decrementa en fin de semana para la mayoría los tramos de edad.
- Sólo se aprecia un ligero incremento en fin de semana en los tramos 17-18 y >66

### Pie user_type por weekend
"""

x_var = 'desc_user_type'
filter_var = 'weekend'
generatePie(x_var, filter_var, dfUsers)

"""Confirmamos el uso mayoritario del tipo de usuario Anual frente a los otros dos tipos de abono. 

Datos de interés:
- La tendencia general es que se usa más en laboral que en finde para los tipos Anual y Company
- Por el contrario el ocasional es usado más el fin de semana, doblando en proporción.

##Viajes

Tras lo visto en los datos con foco en los usuarios, se ve claro y necesario cambiar el foco a los viajes y analizar los mismos por tipo de uso y caracteristicas de usuarios para poder entender el uso y sus diferentes agrupamientos.
"""

dfTrips.describe()

dfTrips.info()

"""En este primer resumen, puede enterse que la media de uso se situa en una franja horaria de mediodia. La velocidad media del viaje indica posibles outliers y un comportamiento muy parejo con la velocidad media del track. Del mismo modo ocurre con las medidas de tiempo medio del viaje y tiempo medio del track. De este modo se entiende que lo más probable con una variable por cada sería suficiente. Para confirmar esta premisa visualizaremos la matriz de correlaciones a continuación.

###Distribución del dataset e indentificación de outliers
"""

generateScatterMatrix([
  'travel_time', 
  'avg_speed', 
  'avg_track_speed', 
  'travel_time_track'
], dfTrips)

"""Como puede entenderse la premisa que se mencionaba sobre la relacion entre ambas variables de velocidad como ambas de tiempo estan correladas. La variable velocidad tiene una correlación muy alta, lo que indica que dichas variables muy similares casi iguales en la información que aportan. Por lo que con una de ellas ya sería suficiente. En el caso del tiempo medio no hay una correlación tan alta, pero si indica de modo que problable que en este caso también con una de ambas variables sea suficiente.

#### Matriz de correlación de variables numéricas

Para confirmar las correlaciones que se ven en los anteriores gráficos, nos disponemos a ejecutar la matriz de correlaciones.
"""

dfCorr = dfTrips[[
    'travel_time', 
    'avg_speed', 
    'avg_track_speed', 
    'travel_time_track'
]].corr(method='pearson')

z = []
for index, row in dfCorr.iterrows():
  z.append(row)
    
generateHeatMap(
    dfCorr.index, 
    dfCorr.index, 
    z, 
    'Correlación Variables Numéricas',
    None,
    None
)

"""Finalmente se confirman las correlaciones anteriormente mencionadas. Por lo que se preservarán en este analisis una varíable por cada par, "travel_time" y "avg_speed"

### Eliminación de los outliers mediante el metodo IQR
"""

columns = ['travel_time', 'avg_speed']
#Transformación a logaritmo para mejorar la distribución normal.
for col in columns:
  dfTrips[col] = np.log(dfTrips[col])

outliers_index = []
for col in columns:
  outliers_index+= identificar_outliers(dfTrips, col)
dfTripsIQR = dfTrips.drop(outliers_index).reset_index(drop = True)

generateScatterMatrix([
  'travel_time', 
  'avg_speed', 
  'avg_track_speed', 
  'travel_time_track'
], dfTripsIQR)

"""Tras la eliminación de los outliers puede observarse como la distribución de las variables se vuelve más Gausiana y los datos ya están mucho mas compactados para poder modelizarlos.

###BoxPlot Code Group

Crearemos una variable "code" con la combinatoria de valores de las distintas variables categóricas que queremos analizar juntas. La idea es analizar las distribuciones de las variables numéricas **travel_time** y **avg_speed** para cada valor de la combinatoria de las variables catégoricas y ver si existen similitudes. Para analizar dichas similitudes, utilizaremos **Kolmogorov-Smirnov** ya que T-Test está más orientado a distribuciones normales. Partiremos del análisis de una única variable e iremos iterando añadiendo cada vez una variable más hasta llegar a todas las variables.
"""



vars = [
  'weekend', 
  'hour_type', 
  'trip_type', 
  'desc_user_type', 
  'desc_ageRange'
]


x_vars = ['travel_time', 'avg_speed']
alpha = 0.05
combi_vars = combinatoria(vars) 
for combi_var in combi_vars:
  if combi_var:
    dfTripsIQR['code'] = getCode(dfTripsIQR, combi_var)
    for x_var in x_vars:
      title = 'BoxPlot ' + '+'.join(combi_var) + '<->' + x_var
      generateBoxPlot(x_var, 'code', dfTripsIQR, title)
      '''
      title = 'T-Test ' + '+'.join(combi_var) + '<->' + x_var
      z_pvalue = []
      values = sorted(dfTripsIQR['code'].unique().tolist())
      #Añadimos un caracter para que no lo transforme a número
      x_values = ['·' + str(x) for x in values] 
      for i in range(len(values)):
        x_pvalue = []
        data1=dfTripsIQR.loc[dfTripsIQR['code']==values[i]][x_var]
        for j in range(len(values)):
          data2=dfTripsIQR.loc[dfTripsIQR['code']==values[j]][x_var]
          t_stat, p = stats.ks_2samp(data1, data2)
          x_pvalue.append(p)
        z_pvalue.append(x_pvalue)
      generateHeatMap(x_values, x_values, z_pvalue, title,
        [[0, '#2E29CD'], 
        [alpha, '#E91160'],
        [1, '#EB1010']
        ],
        dict(
          titleside = 'top',
          tickmode = 'array',
          tickvals = [0, alpha],
          ticktext = ['Diferents','Equals'],
          ticks = 'outside'
        )
      )
      '''

"""Tras analizar detenidamente las distribuciones de los boxplots con las diferentes conbinaciones de variables, se han visto ciertas dependencias o similitudes que pueden llegar a dar una primera visión de las posibles agrupaciones que se van a poder encontrar:

- Se ve una similitud entre los usuarios de edad desconocida y los que están entre 41-65 con bono Anual. Del mismo modo encontramos similitud entre los usuarios de bono Anual y los de edad comprendida entre 19-26 y 27-40.
- Se ve una similitud importante entre las horas de la mañana (7-12 y 12-18) para los usuarios de edad media (27-40 y 41-65). 
- Se ve una similitud importante entre el tipo de usuario Anual y las horas de la mañana (7-12 y 12-18).
- Se ve una similitud importante entre los usuarios ocasionales y de compañia y las horas de la mañana (7-12 y 12-18).
- Se ve una similitud importante entre los usuarios hacen viaje retiro-retiro entre las horas 7-12 y 12-18 y sobre el mismo horario los usuarios que salen del retiro.
- Se ve una similitud importante entre los usuarios que salen del retiro con edades comprendidas entre 19-26 y 27-40. Asi mismo también se ve similitud entre los usuarios que cogen y dejan la bici en la misma estación con edades comprendidas entre 27-40 y 41-65. Además los usuarios de edad desconocida salen del retiro tienen una similitud importante con los usuarios que salen del retiro con edades comprendidas entre 41-65.
- Se ve una similitud importante entre los usuarios con edades comprendidas entre 19-26 y 27-40 en dias laborables. Sobre los mismos días hay una similitud importante con los usuarios de edad desconocida y los de 41-65. Hay similitud entre los usuarios de edades entre 19-26 y 41-65 en el uso de bicimad el fin de semana.
- Se ve una similitud el uso del servicio en dias laborables entre las horas comprendidas de las 7-12 y 12-18.
- Se ve una similitud entre los usuarios viajan retiro-retiro tanto fin de semana como días laborables.

### Tabla contingencias
"""

vars = ['day_of_week', 'weekend', 'hour_type', 'trip_type', 'desc_user_type', 'desc_ageRange']
z_pvalue = []
for varx in range(len(vars)):
  x_pvalue = []
  for vary in range (len(vars)):
    crosstab = pd.crosstab(dfTrips[vars[varx]], dfTrips[vars[vary]])
    cramerv = cramers_v(crosstab)
    if vary > varx:
      generateCrosstabBars(crosstab, vars[varx], vars[vary], True)
    x_pvalue.append(cramerv)
  z_pvalue.append(x_pvalue)

generateHeatMap(
    vars, 
    vars, 
    z_pvalue, 
    'Correlación Variables Categóricas',
    [
      [0, 'rgb(48,0,255)'], 
      [0.3, 'rgb(0, 255, 180)'],
      [0.6, 'rgb(186, 255, 0)'], 
      [1, 'rgb(255,45,0)']
    ],
    dict(
      titleside = 'top',
      tickmode = 'array',
      tickvals = [0, 0.3, 0.6, 1],
      ticktext = ['no dependiente','leve', 'intensa', 'dependiente'],
      ticks = 'outside'
    )
)

"""- Se ve un uso mayoritario en las franjas de uso 7-12 y 12-18.
- El rango de edades que mayor uso hacen son 27-40 y 41-65. 
- Existe un aumento en el uso en las horas entre 7-12 para el rango de edad 41-65.
- Uso mayoritario de usuarios que salen del retiro con bono anual.
- Tienden a crecer los usuarios de edad desconocida el fin de semana.
- Crece el uso el fin de semana en el horario de 18-23 y baja el de 7-12.
- Se duplica el uso el fin de semana en los viajes de la misma estación con bono ocasional.
- Los usuarios de edad desconocida hacen mayor uso en viajes de misma estación.
- Crece el uso de edad desconocida durante las horas de 12-18 y 18-23.
- Hay más uso de usuarios viajan retiro-retiro o se van del retiro que los que hacen el viaje a la misma estación.
- Los usuarios con rango de edad 27-40 hacen un mayor uso en viajes retiro-retiro o saliendo del retiro, disminuyen a la mitad cuando son viajes misma estación.
- El uso de abono ocasional es casi por completo de usuarios con edad desconocida.
- Tras ir analizando todos los datos se ve un uso por usuarios de abono de compañia que denota que añaden a su abono a familiares como se puede ver en el gráfico de las variables por tipo de usuario y rango de edad.

Observando la matriz que muestra la relacion de dependencia que hay entre las variables categoricas, puede observarse por un lado una gran dependencia de las variables day_of_week y weekend. Además se encuentra una dependencia o correlación leve entre las varibles de tryp_type y desc_user_type. Existe una tercera depencia ya más moderada entre las variables de desc_ageRange y desc_user_type.

Por lo tanto las variables categoricas que se van a utilizar para describir el dataset en el modelado son:
- weekend
- hour_type
- trip_type
- desc_user_type
- desc_ageRange

#Clustering

##KMeans
El algoritmo KMeans en Python que viene de scikit-learn tiene el principal hándicap que no acepta variables categóricas. Hay mucha literatura de cómo transformar (one hot encoding, binary, label encoding, etc..) categóricas en numéricas pero cualquiera de ellas puede generar problemas al proveer de ordinalidad a dichas variables. Problema que se acrecenta en KMeans ya que utiliza la distancia euclídea para encontrar los centroides.

En cualquier caso, se realizará un estudio utilizando éste algoritmo. Lo primero que hay que hacer es escalar las numéricas y convertir a númericas las variables categóricas (usaremos la función **get_dummies** (one hot encoding). Utilizar ésta técnica hace que se creen tantas variables como valores tenga cada variable categórica. En consecuencia, parece acertado ejecutar un PCA.

###KMeans One Hot Encoding
"""

#Creacion de Variables: scalar de las numericas y one_hot de las categoricas

vars = ['weekend', 'hour_type', 'trip_type', 'desc_user_type', 'desc_ageRange']
df = pd.get_dummies(dfTripsIQR[vars])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(
    dfTripsIQR[['travel_time', 'avg_speed']]), 
    columns=['travel_time', 'avg_speed']
)

X = pd.concat([df,df_scaled], axis=1)

#PCA
#Fitting the PCA algorithm with our Data
pca = PCA().fit(X)

generateLineChar(
    [None], 
    [np.cumsum(pca.explained_variance_ratio_)], 
    ['X'],
    'PCA Varianza', 
    'Número de componentes', 
    'Varianza (%)'
)

"""Parece que el PCA recomienda un total de 7 componentes.

Nos faltaría encontrar el número de clústers apropiado. También, hay mucha literatura al respecto y no hay **fórmula** adecuada ya que el número de clústers obedece a otras razones como negocio, campaña a realizar, etc..

Una de las técnicas más usadas es **Elbow Curve** en donde se puede apreciar qué número parece recomendable, basado bien en las distancias a los centroides o el scoring calculado.
"""

Nc = range(2, 20)
models = [KMeans(n_clusters=i) for i in Nc]
score = [models[i].fit(X).score(X) for i in range(len(models))]

pca = PCA(n_components=11).fit(X)
Xpca = pca.transform(X)

score_pca = [models[i].fit(Xpca).score(Xpca) for i in range(len(models))]
generateLineChar(
    [[i for i in Nc], [i for i in Nc]], 
    [score, score_pca],
    ['X', 'X_PCA'],
    'Curva Elbow K-Means', 
    'Número de Clusters', 
    'Curva Elbow'
)

"""Sin embargo, usando todas las variables o incluso las variables PCA, la curva no indica un codo claro para discernir el número de clústers."""

Nc = range(2, 20)
models = [KMeans(n_clusters=i) for i in Nc]
score = [models[i].fit(df_scaled).score(df_scaled) for i in range(len(models))]
generateLineChar(
    [[i for i in Nc]], 
    [score],
    ['numerics'],
    'Curva Elbow K-Means Variables Numéricas', 
    'Número de Clusters', 
    'Curva Elbow'
)

"""Usando sólo variables numéricas, parece que el número de clústers es 5. La curva Elbow con variables numéricas es más **bonita** por lo que parece que se empieza a intuir que, quizás, podría realizarse el clustering sólo con las numéricas."""

#KMeans con n=4
n=4
vars = ['weekend', 'hour_type', 'trip_type', 'desc_user_type', 'desc_ageRange']
df = pd.get_dummies(dfTripsIQR[vars])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(
    dfTripsIQR[['travel_time', 'avg_speed']]), 
    columns=['travel_time', 'avg_speed']
)

X = pd.concat([df,df_scaled], axis=1)
kMeans = KMeans(n_clusters=n).fit(X)
dfTripsIQR['cluster'] = kMeans.labels_

vars = ['weekend', 'hour_type', 'desc_user_type', 'desc_ageRange', 'trip_type']
dfTripsCluster = dfTripsIQR[vars + ['cluster']]
x, y, z = generateClusterMap(
    dfTripsCluster, 
    vars, 
    sorted(dfTripsCluster['cluster'].unique().tolist()), 
    'cluster'
  )
title = "Mapa Relación Cluster K-Means One Hot Encoding"
  
generateHeatMap(x, y, z, title, None, None)

"""####Conclusión
Tras analizar el conjunto de resultados con lo visto anteriormente en el analisis explotario, se ve reflejado gran parte de las conclusiones. Se denota gran peso de aquellas variables con un porcentaje muy alto en el dataset. 

Según se ha podido ver en 3 de los clusters parten de una premisa de usuarios con bono anual que salen desde el retiro en días laborables. Les diferencia el rango de edad (41-65, unknown, 27-40) y los rangos horarios donde dos hacen más actividad entre 7-12 y 12-18 y el otro12-18 y 18-23. El cuarto cluster se ve más diferenciado ya que aunque prevalece el tipo de bono y que sale desde el retiro, lo hace en dias laborables y son mayoritariamente usuarios de edad desconocida.

###KMeans con variables numéricas
"""

#KMeans n=4 Variables Numéricas
n=4
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(
    dfTripsIQR[['travel_time', 'avg_speed']]), 
    columns=['travel_time', 'avg_speed']
)
X = df_scaled
kMeans = KMeans(n_clusters=n).fit(X)
dfTripsIQR['cluster'] = kMeans.labels_

"""Aúnque en gran medida los resultados anteriores guardan bastante similitud con lo visto en el analisis exploratorio, lo conveniente sería probar solo con variables númericas como se comporta. Debido a que este algoritmo aplica los calculos teniendo en cuenta la distancia euclidea y por la forma de los datos no vendría a ser la más adecuada."""

vars = ['weekend', 'hour_type', 'desc_user_type', 'desc_ageRange', 'trip_type']
dfTripsCluster = dfTripsIQR[vars + ['cluster']]
x, y, z = generateClusterMap(
    dfTripsCluster, 
    vars, 
    sorted(dfTripsCluster['cluster'].unique().tolist()), 
    'cluster'
  )
title = "Mapa Relación Cluster K-Means Numéricas"
  
generateHeatMap(x, y, z, title, None, None)

"""####Conclusión
Tal y como se ha podido comprobar gran parte de las variables más descriptoras se han mantenido. Aunque en este caso se parte de una base en la que los cuatro clusters serian viajes de usuarios con bono anual durante la semana que salen del retiro. En este caso también lo que varia son algo las edades aunque hace mucho foco en los de edad desconocida y los rangos horarios. Tiene similitudes con las agrupaciones que se veian en el analisis exploratorio pero no hay variedad de agrupaciones, ya que se ven todos muy cerca. 

De modo que este metodo no sería del todo definitivo a la hora de poder describir el dataset tal y como se ha entendido a lo largo de este estudio. Para seguir concretando se probara con el modelo Gaussian Mixture

##Gaussian Mixture
Este algorito tiene la misma problematica que el KMeans con las variables categóricas, ademas de necesitar que tengan una distribución gausiana.

Aún asi, se va a disponer a probar los mismos casos que con el modelo anterior para poder hacer comparaciones en las coclusiones de cada uno de los casos.

###Gaussian Mixture con One Hot Encoding
"""

vars = ['weekend', 'hour_type', 'trip_type', 'desc_user_type', 'desc_ageRange']
df = pd.get_dummies(dfTripsIQR[vars])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(
   dfTripsIQR[['travel_time', 'avg_speed']]), 
   columns=['travel_time', 'avg_speed']
)

X = pd.concat([df,df_scaled], axis=1)
pca = PCA(n_components=11).fit(X)
Xpca = pca.transform(X)

#Para encontrar el "codo" en GaussianMixture, se usan los valores Akaike
#https://es.wikipedia.org/wiki/Criterio_de_informaci%C3%B3n_de_Akaike
Nc = range(2, 20)
models = [GaussianMixture(
    n_components=i, 
    covariance_type='full', 
    random_state=45) for i in Nc]
models_fit = [models[i].fit(X) for i in range(len(models))]
score_bic = [models_fit[i].bic(X) for i in range(len(models))]
score_aic = [models_fit[i].aic(X) for i in range(len(models))]

pca = PCA(n_components=11).fit(X)
Xpca = pca.transform(X)

models_fit = [models[i].fit(Xpca) for i in range(len(models))]
score_pca_bic = [models_fit[i].bic(Xpca) for i in range(len(models))]
score_pca_aic = [models_fit[i].aic(Xpca) for i in range(len(models))]

generateLineChar(
    [[i for i in Nc], [i for i in Nc], [i for i in Nc], [i for i in Nc]], 
    [score_bic, score_pca_bic, score_aic, score_pca_aic],
    ['X BIC', 'X_PCA BIC', 'X AIC', 'X_PCA AIC'],
    'Curva Elbow GaussianMixture', 
    'Número de Clusters', 
    'Curva Elbow'
)

"""Para este caso también se va a tratar de ver si la curva de Elbow es indicador de que número de clusters ha de tener. 

Para poder pintar dicha curva se utiliza la métrica de AIC, maneja un solución de compromiso entre la bondad de ajuste del modelo y la complejidad del modelo. Este se basa en ofrecer una estimación relativa de la información perdida cuando se utiliza un modelo determinado para representar el proceso que genera los datos.

Tal como puede observarse en el gráfico de arriba, este tampoco es determinante para poder tomar una decisión del número de clusters basado en dicha métrica. Por lo tanto se procede a hacer un clustering al igual que en el modelo anterior, con 4 (que son los que se han ido deduciendo durante el analisis).
"""

#GaussianMixture con n=4
n=4
vars = ['weekend', 'hour_type', 'trip_type', 'desc_user_type', 'desc_ageRange']
df = pd.get_dummies(dfTripsIQR[vars])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(
    dfTripsIQR[['travel_time', 'avg_speed']]), 
    columns=['travel_time', 'avg_speed']
)

X = pd.concat([df,df_scaled], axis=1)
gaussianMixture = GaussianMixture(
    n_components=n, 
    covariance_type='full', 
    random_state=45).fit(X)
dfTripsIQR['cluster'] = gaussianMixture.predict(X)

vars = ['weekend', 'hour_type', 'desc_user_type', 'desc_ageRange', 'trip_type']
dfTripsCluster = dfTripsIQR[vars + ['cluster']]
x, y, z = generateClusterMap(
    dfTripsCluster, 
    vars, 
    sorted(dfTripsCluster['cluster'].unique().tolist()), 
    'cluster'
  )
title = "Mapa Relación Cluster GaussianMixture One Hot Encoding"
  
generateHeatMap(x, y, z, title, None, None)

"""####Conclusión
Como se puede observar tiene muchas similitudes con el resultado del KMeans con One Hot Encoding. Se ven cuatro clusters definidos por un mismo patros que luego varía según el rango de edad, muy marcado, y un rango horario. Se ven dos clusters de usuarios que con bono anual salen del retiro entre las 7-12 y 12-18 con rangos de edad que cuadrán totalmente tanto con anteriores modelos como lo visto en el analisis (27-40 y 41-65).

Lo mas diferencial en este caso son los usuarios que salen del retiro el fin de semana con bono anual y con un rango de edad desconocido.

###Gaussian Mixture con variables numericas

Se procede a probar el modelo bajo las mismás características que el anterior para poder comparar los resultados.
"""

#GaussianMixture n=4 Variables Numéricas
n=4
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(
    dfTripsIQR[['travel_time', 'avg_speed']]), 
    columns=['travel_time', 'avg_speed']
)
X = df_scaled
gaussianMixture = GaussianMixture(
    n_components=n, 
    covariance_type='full', 
    random_state=45).fit(X)
dfTripsIQR['cluster'] = gaussianMixture.predict(X)

vars = ['weekend', 'hour_type', 'desc_user_type', 'desc_ageRange', 'trip_type']
dfTripsCluster = dfTripsIQR[vars + ['cluster']]
x, y, z = generateClusterMap(
    dfTripsCluster, 
    vars, 
    sorted(dfTripsCluster['cluster'].unique().tolist()), 
    'cluster'
  )
title = "Mapa Relación Cluster GaussianMixture Numéricas"
  
generateHeatMap(x, y, z, title, None, None)

"""####Conclusión
Este resultado quizás sea el menos descriptor de todos los modelos que se han probado hasta el momento. Mantiene un patron muy marcado por el bono anual, salida desde el retiro y en dias laborables en un rando de edad mayoritariamente desconocido. 

Por lo que no se puede tomar este resultado como que describe en un alto porcentaje el dataset

##KModas
Tras ver dos modelos, los cuales muchas teorias indican que están más bien preparados para hacer cluster con variables numéricas. Y visto los resultados, se quiere contrastar con un modelo que sigue el mismo patron pero si esta preparado para entender variables categoricas.

Este modelo define los grupos en función del número de categorías coincidentes entre los puntos de datos.
"""

#KModes n=4. Sin necesidad de categorizar variables..


n=4
vars = [
    'travel_time', 
    'avg_speed', 
    'weekend', 
    'hour_type', 
    'trip_type', 
    'desc_user_type', 
    'desc_ageRange'
]

X = dfTripsIQR[vars]
kMode = KModes(n_clusters=n, init='Huang').fit(X)
dfTripsIQR['cluster'] = kMode.labels_

vars = ['weekend', 'hour_type', 'desc_user_type', 'desc_ageRange', 'trip_type']
dfTripsCluster = dfTripsIQR[vars + ['cluster']]
x, y, z = generateClusterMap(
    dfTripsCluster, 
    vars, 
    sorted(dfTripsCluster['cluster'].unique().tolist()), 
    'cluster'
  )
title = "Mapa Relación Cluster KModes"
  
generateHeatMap(x, y, z, title, None, None)

"""####Conclusión
El resultado del modelo confirma los resultados que se han obtenido en las pruebas anteriores. Denotan cuatro clusters bastante cercanos entre si con un patron definido por bono anual y salida desde el retiro. Donde en este caso si pierde algo de fuerza la variable de dia laboral aunque sigue siendo muy fuerte y dando como resultado 3 clusters en dia laborable y tan solo uno en fin de semana. En este caso toman un peso fuerte y decisivo los rangos horarios de 12-18 y 18-23. En cuanto al rango de edad la gran mayoria se situan en rango desconocido y tan solo uno señala 27-40 como edad descriptora con gran peso.

Por lo tanto es sencillo decir que sigue la misma linea descriptora que los anteriores modelos aún y este estar preparado para variables categoricas y en los otros ser más discutible.

##SNA

Se parte de la idea de comparar las distribuciones a través del **Code Group**, de tal manera que cada valor del **Code** se considera como un nodo de una red. A partir del **Kolmogorov-Smirnov**, si dos Codes tienen la misma distribución, podemos suponer que están unidos con un determinado peso proporcionado por el p_value. Una vez creada la red, a partir de los algoritmos de comunidad, se obtendrán todos los nodos que pertenecen a la misma comunidad, en definitiva, al mismo clúster.

Se generará, por tanto, una red en donde dos nodos estará unidos sí su distribución es igual (p_value>=alfa, siendo alfa 0.05)

Dado que existen dos variables númericas y, en consecuencia, dos distribuciones para cada code, disponemos de varios p_values para cada code, en consecuencia, será necesario **inventar** el peso asociado a los dos nodos de la red.

Una primera aproximación sería utilizando la media de los valores. Otra aproximación podría ser agregando las variables númericas en una sola y calcular los p_value sobre ella, por ejemplo, la variable **travel_space_avg = travel_time * avg_speed**.
"""

#Generacion de un dataframe con los p_values
combi_var = [
  'weekend', 
  'hour_type', 
  'trip_type', 
  'desc_user_type', 
  'desc_ageRange'
]


dfTripsIQR['code'] = getCode(dfTripsIQR, combi_var)
x_vars = ['travel_time', 'avg_speed', 'travel_space_avg']
#dfNodes = generateRelationPValue(combi_var, x_vars, dfTripsIQR)
dfNodes = pd.read_csv(file + '.nodes', sep=';')
dfNodes.head()

#Inicializacion de variables Networks
Gs = {
    'mean': {'G': nx.Graph(), 'partition': None},
    'space': {'G': nx.Graph(), 'partition': None}
}

"""### Media travel_time y avg_speed"""

#Creacion de la red utilizando como peso la media de ambos p_values
alpha = 0.05
for index, row in dfNodes.iterrows():
  p = (row['p_travel_time'] + row['p_avg_speed'])/2
  if p> alpha:
    Gs['mean']['G'].add_edge(row['nodeA'], row['nodeB'], weight=p)
nx.write_gml(Gs['mean']['G'], file + '.mean.gml')

"""### Uso de variable travel_space_avg"""

#Creacion de la red utilizando como peso la media de ambos p_values
alpha = 0.05
for index, row in dfNodes.iterrows():
  p = row['p_travel_space_avg']
  if p> alpha:
    Gs['space']['G'].add_edge(row['nodeA'], row['nodeB'], weight=p)
nx.write_gml(Gs['space']['G'], file + '.space.gml')

"""### Información de la Red"""

for g in Gs:
  #G=nx.read_gml(, file + '.' + g + '.gml')
  G = Gs[g]['G']
  print("Network " + g + "----------------------------------------")
  print('nodes: ',G.order(),'; size: ',G.size())


  print("eccentricity: %s" % nx.eccentricity(G))
  print("radius: %d" % nx.radius(G))
  print("diameter: %d" % nx.diameter(G))
  print("center: %s" % nx.center(G)) #Nodos que estan en el centro (que estan a distancia radius)
  print("periphery: %s" % nx.periphery(G)) #Nodos mas alejados (que estan a distancia diameter)
  print("density: %s" % nx.density(G))
  print("---------------------------------------------------------")

"""### Generación de comunidades"""

for g in Gs:
  G = Gs[g]['G']
  Gs[g]['partition']=co.community_louvain.best_partition(G) 
  print("Network " + g + '->', 'nodes: ', G.order(),
    '; communities: ',len(set(Gs[g]['partition'].values())))

vars = ['weekend', 'hour_type', 'desc_user_type', 'desc_ageRange', 'trip_type']
for g in Gs:
  partition = Gs[g]['partition']
  dfComs=pd.DataFrame({
      'node':list(partition.keys()),
      'com':list(partition.values())
  })
  dfTripsCluster = dfTripsIQR.merge(
      dfComs, 
      left_on='code', 
      right_on='node', 
      how='inner'
  )
  x, y, z = generateClusterMap(
    dfTripsCluster, 
    vars, 
    dfTripsCluster['com'].unique().tolist(), 
    'com'
  )
  title = "Mapa Relación Cluster SNA"
  if g=='mean':
    title+=": Media travel_time y avg_speed"
  else:
    title+=": travel_space_avg"
  generateHeatMap(x, y, z, title, None, None)

"""###Conclusión
Tal y como puede verse en los resultados, estos son donde los clusters estan más definidos y con una mayor distancia entre ellos. De cara a una posible predicción estaría mucho más claro a cual debería ir el nuevo viaje. 

En el primero de los casos, donde se calcula la media de los valores de ambas variables, indica tal y como se viene observando hasta ahora cuatro clusters. En este caso a diferencia del resto a sido totalmente inducido por el modelo. Donde diferencia fuertemente por usuarios con bono anual o de compañia, y si sale desde el retiro o no. 

Por lo tanto estos clusters serian:
- Viajes en días laborables con bono anual con edades comprendidas entre 27-40 y desconocidas. En unos horarios de 12-18 sobre todo.
- Viajes en días laborables con una mayoria de bono de compañia con una edad comprendida entre 27-40 en horario de 7-12 y 12-18.
- Viajes en días laborables mayoritariamente con bono anual y edad desconocida en un horario mucho más disperso y equilibrado.
- Usuarios con bono de compañia que salen y dejan la bici en la misma estación en un horario de 7-12 con edades comprendidas entre 27-40 y 41-65.

Teniendo en cuenta la otra alternativa con travel_space_avg, el resultado se reduce a 3 clusters. Estos están más cerca que los anteriores y se intuye que no describen el mismo porcentaje que el anterior, si no menos. Dos de los clusters son bien similares al anterior resultado y el tercero en una variación donde son usuarios que viajan desde el retiro en día laboral con abono ocasional y edad desconocida en un horario muy marcado de 12-18.

##Conclusión

Tras analizar el dataset, y ver como se han comportado todos los modelos se llegan a comprender varias cosas.
- El analisis inicial se encuentra representado en gran media por gran parte de los modelos.
- Las variables categoricas incluso en modelos discutidos por su inclusión han aportado información que ha ayudado a describir mejor el dataset.
- Con todo ello es cierto que hay parte quizás por el bajo volumen de samples que no llegan a estar quizás debidamente representados.
- El modelo de SNA es quién describe mejor mayor parte del dataset sin dejar de representar lo intuido en el analisis.

En cuanto al resultado en total de todos los modelos se pueden inferir los siguientes dos grupos más definidos y otros dos algo no tan claros:
- Viajes de usuarios entre 27-40 y 41-65 que salen desde el retiro en un horario de 7-12 y 12-18 con bono anual.
- Viajes de usuarios entre 41-65y rango desconocido que salen en horario de 12-18 y 18-23 con bono en mayor medida anual.
- Viajes de usuarios de fin de semana con bono anual mayoritariamente y en un rango horario comprendido entre 12-18 y 18-23.
- Viajes de usuarios de compañia en días laborables y rangos mayoritariamente entre 27-40 y 41-65.

Estos dos últimos estarían menos claros su definición, lo que termina dificultando el tener un dibujo definitivo del dataset. Por lo que a este estudio le tendrían que seguir acciones futuras para tratar de comprender mejor los datos. 
- Habría que investigar que señal darían los datos si se quitase el rango de edad Unknown que es un alto porcentaje y directamente no esta aportando demasiada información. 
- Además habría que valorar que resultados salen restandole peso descriptivo con tecnicas de balanceo o incluso quitando del dataset a la variable desc_user_type que contiene más de un 90% de samples de bono anual.
- Se valoraría agregar más la variable de rango de edad ya que las franjas no estan niveladas y quizás con menos se obtendría mejor señal.
- Se valoraría estudiar más modelos que manejen variables categoricas para ver si dictan resultados más claros.
"""