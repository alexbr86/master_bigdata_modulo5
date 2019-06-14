#Inventario de funciones
import pandas as pd
import numpy as np
'''
  Función para calcular geo distancias.
  Recibe la longitud y latitud de dos puntos
  Devuelve la distancia.
'''
def calculateDistance(lat1, lon1, lat2, lon2):
  from math import sin, cos, sqrt, atan2, radians
  
  R = 6373.0 # approximate radius of earth in km

  lat1 = radians(lat1)
  lon1 = radians(lon1)
  lat2 = radians(lat2)
  lon2 = radians(lon2)

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))
  return R * c


'''
  Función para calcular velocidades basado en la información de los Tracks del viaje.
  Recibe el DataFrame de Viajes con los Tracks
  Quitaremos los tramos en los que speed = 0
  Devuelve un DataFrame con:
    oid => Identificador del viaje
    travel_time_track => Suma de Tiempos
    avg_speed_track => Calculado a través de la distancia total / tiempo total
    avg_speed => Media de los valores speed
'''
def calculateSpeed(df):
  import datetime
  print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), 'Ini')
  oids = df['oid'].unique()
  df = pd.DataFrame(columns=['oid', 'travel_time_track', 'avg_speed_track', 'avg_speed'])
  for oid in oids:
    s = 0.0
    t = 0.0
    seconds_ant = 0.0
    v_avg = 0.0
    n_avg = 1
    dfTracks = df.loc[df['oid']==oid][['secondsfromstart', 'speed']].sort_values(by=['secondsfromstart']).reset_index()
    for index, row in dfTracks.iterrows():
      if row['speed']>0.0:#Eliminamos velocidades 0
        diff_t = row['secondsfromstart']-seconds_ant
        s+= row['speed']*diff_t
        t+= diff_t
        v_avg+=row['speed'] #v_avg_n = ((n-1)*v_avg_n-1 + v)/n
        n_avg+=1
      seconds_ant = row['secondsfromstart']
    if t==0.0: #Es posible que no haya tracks
      v = 3000.0 #velocidad absurda para tenerlos identificados
      v_avg = 3000.0 #velocidad absurda para tenerlos identificados
    else:
      v = s/t
      v_avg = v_avg/(n_avg-1)
    df = pd.concat([df, pd.DataFrame({
      'oid': [oid],
      'travel_time_track': [t], 
      'avg_speed_track': [v], 
      'avg_speed': [v_avg]
    })])
  print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), 'Fin')
  # Donde no haya valor de speed (3000), ponemos el valor medio.
  vars_mean = {'avg_track_speed':3000.0, 'avg_speed':3000.0, 'travel_time_track':0.0}
  query = ' & '.join([f'{k}!={v}' for k, v in vars_mean.items()])
  dfAux = dfSpeeds.query(query)
  dfAux = dfAux[[k for k, v in vars_mean.items()]].mean().reset_index()
  for k, v in vars_mean.items():
    dfSpeeds[k] = pd.np.where(
      dfSpeeds[k]==v, dfAux.loc[dfAux['index']==k, 0], 
      dfSpeeds[k]
    )
  return df

'''
  Funcion que añade variables extras como Fin de Semana, Horario, etc
  Recibe el Dataframe de Viajes
  Devuelve tantas variables como se hayan definido:
    day_of_week => Literal dia de la semana.
    weekend => Si el día es Fin de semana (weekend) o laboral (working day)
    hour_of_day => Hora del día (0-23)
    hour_type => Agrupamiento de horas en el día.
    trip_type => Agrupamiento de las estaciones de enganche y desenganche:
      Misma Estación, Retiro (origen y destino es en Retio), a y desde Retiro.
    
'''
def addFeatures(df, retiro_stations):
  def getHourType(hour):
    if hour>=23:
      return "(23:00-07:00)"
    if hour<8:
      return "(23:00-07:00)"
    if hour<13:
      return "(07:00-12:00)"
    if hour<19:
      return "(12:00-18:00)"
    if hour<23:
      return "(18:00-23:00)"
    
  def getTripType(id_unplug, id_plug, retiro_stations):
    if id_unplug==id_plug:
      return "Same Station"
    if id_unplug in retiro_stations:
      if id_plug in retiro_stations:
        return "Retiro"
      return "From Retiro"
    return "To Retiro"
     
  df['daysample'] = pd.to_datetime(df['daysample'])
  df['day_of_week'] = df['daysample'].dt.dayofweek
  # 5 => Sat, 6 =>Sun, 0 => Mon, 1 => Tue...
  df['weekend'] = pd.np.where(
      df['day_of_week']>4, 'Weekend', 'Laboralday'
  )
  
  df['datesample'] = pd.to_datetime(df['datesample'])
  df['hour_of_day'] = df['datesample'].dt.hour
  df['day_of_week'] = '(' + df['daysample'].dt.dayofweek.astype('str') 
  df['day_of_week']+= ') ' + df['daysample'].dt.day_name()  
  df['hour_type'] = [getHourType(hour) for hour in df['hour_of_day']]
  df['trip_type'] = [getTripType(id_unplug, id_plug, retiro_stations) 
    for id_unplug, id_plug in 
      zip(df['idunplug_station'], df['idplug_station'])
  ]
  return df
'''
  Función que devuelve un código que identifica al viaje 
  basado en un pool de variable.
  Recibe el DataFrame de Viajes y las variables
  Devuelve el Dataframe con la variable 'code'
'''
def getCode(df, vars):
  dfAux = pd.DataFrame({'code':['' for i in range(df.shape[0])]})
  for var in vars:
    if var in ['weekend', 'trip_type', 'desc_user_type']:
      dfAux['code']+=df[var].str.slice(0, 1)
    elif var=='hour_type':
      dfAux['code']+=df[var].str.slice(1, 3)
    elif var=='desc_ageRange':
      dfAux['code']+=df[var].str.slice(0, 2)
    elif var=='day_of_week':
      dfAux['code']+=df[var].str.slice(4, 2)
  return dfAux['code']

'''
  Función que identifica los outliers por columna
  Copia pega de la vista en clase con Rafa.
'''
def identificar_outliers(df, col_name):
  #q1, q3 = np.percentile(df[col_name], [25, 75])
  q1, q3 = np.quantile(df[col_name], [0.25, 0.75])
  step = 1.5*(q3-q1)
  mask = df[col_name].between(q1 - step, q3 + step, inclusive=True) #identifica los que estan dentro
  iqr = df.loc[~mask].index #negado de la mascara que hemos aplicado.
  return list(iqr)

'''
  Función que Calcula y devuelve el conjunto potencia de la lista c.
'''
def combinatoria(c):
  if len(c) == 0:
    return [[]]
  r = combinatoria(c[:-1]) 
  r = r + [s + [c[-1]] for s in r]
  return [e for e in sorted(r, key=lambda s: (len(s), s))]

'''
  Función que Calcula el coeficiente de cramer_v.
'''
def cramers_v(confusion_matrix):
  import scipy.stats as sc
  chi2 = sc.chi2_contingency(confusion_matrix)[0]
  n = confusion_matrix.sum().sum()
  phi2 = chi2/n
  r,k = confusion_matrix.shape
  phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
  rcorr = r-((r-1)**2)/(n-1)
  kcorr = k-((k-1)**2)/(n-1)
  return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

'''
  Función que genera relación entre nodos con el p_value utilizando ks_2samp
  Para cada tupla de valores proporcionados por las variables combi_var
  calcula si la distribución de cada variable en vars es igual.
'''
def generateRelationPValue(combi_var, vars, df):
  x_vars = {}
  for var in vars:
    x_vars[var] = pd.DataFrame(columns=['nodeA', 'nodeB', 'p_' + var])
  values = sorted(df['code'].unique().tolist())
  for x_var in x_vars:
    for i in range(len(values)):
      data1=dfTripsIQR.loc[dfTripsIQR['code']==values[i]][x_var]
      for j in range(i+1, len(values)):
        data2=dfTripsIQR.loc[dfTripsIQR['code']==values[j]][x_var]
        t_stat, p_value = stats.ks_2samp(data1, data2)
        #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), x_var, values[i], values[j], t_stat, p_value)
        p_var = 'p_' + x_var
        x_vars[x_var] = pd.concat([x_vars[x_var], pd.DataFrame({
          'nodeA': [values[i]],
          'nodeB': [values[j]],
          p_var: [p_value]       
        })])
  dfNodes = pd.DataFrame()
  for x_var in x_vars:
    if dfNodes.empty:
      dfNodes = x_vars[x_var]
    else:
      dfNodes = dfNodes.merge(x_vars[x_var], 
        left_on=['nodeA', 'nodeB'],
        right_on=['nodeA', 'nodeB'],
        how='inner'
      )
  return dfNodes

'''
  Función que genera la tupla necesaria para renderizar un HeatMap
  de importacion de variables sobre un cluster
  df => dataframe junto con una variable que indica el cluster al que pertenece.
  vars => variables implicadas
  cluster => listado de clusters
  cluster_var => el nombre de la variable del dataframe que tiene 
  el valor del cluster
'''
def generateClusterMap(df, vars, clusters, cluster_var):
  total_registers = df.shape[0]
  dfWeightGlobal = pd.DataFrame(columns=['label', 'weight_global'])
  for var in vars:
    dfAmount = df.groupby(var)[var].count().to_frame()
    dfAmount.columns = ['weight_global']
    dfAmount['label'] = [var + '_' + value for value in dfAmount.index]
    dfAmount['weight_global'] = dfAmount['weight_global']/total_registers
    dfAmount = dfAmount.reset_index()
    dfAmount = dfAmount.drop(columns=[var])
    dfWeightGlobal = pd.concat([dfWeightGlobal, dfAmount], sort=True)
  dfWeightGlobal = dfWeightGlobal.sort_values('label')
  z = []
  for cluster in clusters:
    dfCluster = df.loc[df[cluster_var]==cluster][vars]
    total_registers = dfCluster.shape[0]
    dfWeightCluster = pd.DataFrame(columns=['label', 'weight_cluster'])
    for var in vars:
      dfAmount = dfCluster.groupby(var)[var].count().to_frame()
      dfAmount.columns = ['weight_cluster']
      dfAmount['label'] = [var + '_' + value for value in dfAmount.index]
      dfAmount['weight_cluster'] = dfAmount['weight_cluster']/total_registers
      dfAmount = dfAmount.reset_index()
      dfAmount = dfAmount.drop(columns=[var])
      dfWeightCluster = pd.concat([dfWeightCluster, dfAmount], sort=True)
    dfWeightCluster = dfWeightCluster.merge(
      dfWeightGlobal, 
      left_on='label', 
      right_on='label', 
      how='outer', 
      indicator=True
    )
    dfWeightCluster['weight_cluster'] = pd.np.where(
      dfWeightCluster['_merge']=='both', 
      dfWeightCluster['weight_cluster'], 
      0.0
    )
    dfWeightCluster = dfWeightCluster.drop(columns=['_merge'])
    dfWeightCluster = dfWeightCluster.sort_values('label')
    #z.append((dfWeightCluster['weight_cluster']*dfWeightCluster['weight_global']).to_list())
    z.append(dfWeightCluster['weight_cluster'].to_list())
  x = dfWeightGlobal['label'].to_list()
  y = ['cluster: ' + str(cluster) for cluster in clusters]
  return x, y, z
  
  