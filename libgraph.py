

"""## Funciones para graficar"""

#Inventario de funciones para Graficar => Plotly
colors=['#09BB9F','#15607A','#FFD7E9','#1D81A2','#EB89B5','#18A1CD','#879DC6']

# Método para que funcione Plotly en colab
def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  from plotly import tools
  #Descomentar esta linea en colab
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  tools.set_credentials_file(
        username='jazzphoenix', 
        api_key='euOGc1f62ScWry0fPG8D'
  )
  init_notebook_mode(connected=True)

#Método para generar histogramas de forma dinámica
def generateHistogram(filter_var, x_var, df, cols):
  from plotly.offline import iplot
  import plotly.graph_objs as go
  from plotly import tools

  enable_plotly_in_cell()
  uniq_datas = sorted(df[filter_var['name']].unique().tolist())
  rows = len(uniq_datas) // cols + len(uniq_datas) % cols
  fig = tools.make_subplots(rows=rows, cols=cols, print_grid=False)
  i = 1
  j = 1
  k = 1
  for uniq_data in uniq_datas:
    #histnorm='percent', 'probability'
    trace = go.Histogram(
      x = df.loc[df[filter_var['name']]==uniq_data, x_var],
      name = filter_var['desc'] + ' ' + uniq_data,
      marker = dict(
        color=colors[i-1],
        line = dict(width = 0.5, color = "black")
      ),
      opacity=0.75,
    )
    fig.append_trace(trace, j, k)
    '''
    trace2 = go.Histogram(
      x = df.loc[df[filter_var['name']]==uniq_data, x_var]/2,
      name = filter_var['desc'] + ' 2 ' + uniq_data,
      marker = dict(
        color=colors[i],
        line = dict(width = 0.5, color = "black")
      ),
      opacity=0.75,
    )
    fig.append_trace(trace2, j, k)
    '''
    fig['layout']['xaxis' + str(i)].update(title=x_var)
    fig['layout']['yaxis' + str(i)].update(title='total users')
    i+=1
    if k==cols:
      k = 0
      j+= 1
    k+=1

  fig['layout'].update(showlegend=True, 
    title='Histogram ' + x_var + ' by ' + filter_var['desc'],
    height = 400*rows,
  )
  iplot(fig, filename='histogram_'+ x_var + '_' + filter_var['desc'])
  
#Método para generar un quesito de forma dinámica
def generatePie(x_var, filter_var, df):
  from plotly.offline import iplot
  import plotly.graph_objs as go

  enable_plotly_in_cell()
           
  filter_values = ['Total register']
  if filter_var!='':
    filter_values+= df[filter_var].unique().tolist()
  labels = sorted(df[x_var].unique().tolist())
  i=0
  annotations = []
  data = []
  for filter_value in filter_values:
    values = []
    for label in labels:
      if filter_value=='Total register':
        values.append(df.loc[df[x_var]==label].shape[0])
      else:
        values.append(df.loc[(df[x_var]==label) & (df[filter_var]==filter_value)].shape[0])
    trace = {
      'labels': labels,
      'values': values,
      'type': 'pie',
      'name': x_var,
      'hole': 0.4,
      'domain': {'column': i}
    }
    data.append(trace)
    annotation = {
      "font": {"size": 16},
      "showarrow": False,
      "text": filter_value,
      "x": 0.11 + 1.2*i/len(filter_values),
      "y": 0.5
    }
    annotations.append(annotation)
    i+=1
    fig = {
      'data': data, 
      'layout': {
        'title': x_var,
        "grid": {"rows": 1, "columns": len(filter_values)},
        "annotations": annotations
    }}
  iplot(fig, filename='pie_' + x_var + '_' + filter_var)

#Método para generar gráficos BoxPlot de forma dinámica
def generateBoxPlot(x_var, filter_var, df, title):
  from plotly.offline import iplot
  import plotly.graph_objs as go

  enable_plotly_in_cell()
  values = df[filter_var].unique().tolist()
  data = []
  for value in values:
    #Añadimos un caracter para que no lo transforme a número
    trace = go.Box(
      y=df.loc[df[filter_var]==value][x_var],
      name='·' + str(value),
      notched=True,
      boxmean='sd'
    )
    data.append(trace)
  layout = go.Layout(
    title = title
  )
  fig = go.Figure(data=data,layout=layout)    
  iplot(fig)

#Método para generar Scatter Matrix de forma dinámica 
def generateScatterMatrix(vars, df):
  from plotly.offline import iplot
  import plotly.graph_objs as go
  from plotly import tools
  import plotly.figure_factory as ff

  enable_plotly_in_cell()
  fig = tools.make_subplots(rows=len(vars), cols=len(vars), print_grid=False)
  k=1
  for i in range(len(vars)):
    for j in range(len(vars)):
      varx = vars[i]
      vary = vars[j]
      if varx==vary:
        # Create distplot with curve_type set to 'normal'
        fig_plot = ff.create_distplot(
            [df[varx]], 
            [varx], 
            bin_size=.5, 
            colors=['#15607A']
        )
        dist_plot=fig_plot['data']        
        fig.append_trace(dist_plot[0], i+1, j+1)
        fig.append_trace(dist_plot[1], i+1, j+1)
      else:
        trace = go.Scattergl(
          x = df[varx],
          y = df[vary],
          mode = 'markers',
          marker = dict(line = dict(width = 1, color = '#15607A'))
        )
        fig.append_trace(trace, i+1, j+1)
      fig['layout']['xaxis' + str(k)].update(title=varx)
      if varx!=vary:
        fig['layout']['yaxis' + str(k)].update(title=vary)
      k+=1
  fig.layout.update(showlegend=False, 
    title='Scatter Matrix Variables Numéricas',
    height = 300*len(vars), width=1000
  )

  iplot(fig, filename='scatter_matrix')  

#Método para generar HeatMap
def generateHeatMap(x, y, z, title, colorscale, colorbar):
  from plotly.offline import iplot
  import plotly.graph_objs as go

  enable_plotly_in_cell()
  
  trace = go.Heatmap(
      z=z,
      x=x,
      y=y,
      xgap = 3,
      ygap = 3,
      colorscale = colorscale,
      colorbar = colorbar
  )
  data=[trace]
  layout = go.Layout(
    title = title,     
    margin=go.layout.Margin(l=100, b=200),
    yaxis={'tickangle':-45}
    
  )
  fig = go.Figure(data=data, layout=layout)
  iplot(fig, filename='labelled-heatmap') 
  
  
#Método para pintar la relación de correlación entre dos variables 
def generateCrosstabBars(crosstab, xvar, yvar, percentage):
  from plotly.offline import iplot
  import plotly.graph_objs as go
  from plotly import tools
  enable_plotly_in_cell()
  
  fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
  data = []
  x = crosstab.index.tolist()
  if percentage:
    columnsPercertage = []
    for i in range(len(x)):
      columnsPercertage.append(sum(crosstab.iloc[i, :].tolist()))
  for i in range (crosstab.columns.shape[0]):
    if percentage:
      a = crosstab.iloc[:, i].tolist()
      b = columnsPercertage
      c = [str(round(100*x/y,2))+'%' for x, y in zip(a, b)]
    else:
      c = crosstab.iloc[:, i].tolist()
    trace = go.Bar(
      x=x,
      y=c,
      name=str(crosstab.columns[i])
    )
    data.append(trace)

  layout = go.Layout(
      barmode='group',
      title = 'CrossTab ' + xvar + '<->' + yvar,
      xaxis = dict(title = xvar),
      yaxis = dict(title = yvar)
  )

  fig = go.Figure(data=data, layout=layout)
  iplot(fig, filename='grouped-bar')

def generateLineChar(x, y, names, title, xtitle, ytitle):
  from plotly.offline import iplot
  import plotly.graph_objs as go
  enable_plotly_in_cell()
  data = []
  for i in range(len(x)):
    if x[i]:
      trace = go.Scatter(x = x[i], y = y[i], name = names[i])
    else:
      trace = go.Scatter(y = y[i], name = names[i])
    data.append(trace)
  layout = dict(title = title,
    xaxis = dict(title = xtitle),
    yaxis = dict(title = ytitle),
  )
  fig = dict(data=data, layout=layout)
  iplot(fig, filename='basic-line')
