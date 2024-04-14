import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from dash import html,dcc,Dash
from dash.dependencies import Input , Output 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px




df = pd.read_csv("prepared_data.csv")
df = df.drop(['Unnamed: 0'],axis=1)

tsne = TSNE(n_components=2,  perplexity=30, learning_rate=0.1, n_iter=2000)
X_tsne = tsne.fit_transform(df)
df_tsne = pd.DataFrame(data=X_tsne, columns=['x', 'y'])

def Agglomerative(n_clusters, linkage_type, metric_type, data):
    Agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type, metric=metric_type)
    cluster_labels = Agg.fit_predict(data)
    fig_scatter = px.scatter(df_tsne, x="x", y="y", color=cluster_labels)
    fig_scatter.update_layout(title="Agglomerative Hierarchical Clustering")
    return fig_scatter

fig = Agglomerative(3 , 'single'  ,'euclidean' , df)

def dbscan(eps, no_samples, data):
    db_scan = DBSCAN(eps=eps, min_samples=no_samples).fit(data)
    labels = db_scan.labels_
    scatter_fig = px.scatter(df_tsne, x="x", y="y", color=labels)
    scatter_fig.update_layout(title="DBSCAN Clustering")
    return scatter_fig

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    scatter_fig = px.scatter(df_tsne, 'x', 'y', color=labels, size_max=10)
    scatter_fig.update_layout(title="KMeans Clustering")
    return scatter_fig


app = Dash()
server = app.server
app.layout = html.Div([
    html.H1("Clustering Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    dcc.Dropdown(
        id='algorithm-dropdown',
        options=[
            {'label': 'Agglomerative Hierarchical Clustering', 'value': 'agglomerative'},
            {'label': 'DBSCAN', 'value': 'dbscan'},
            {'label': 'KMeans', 'value': 'kmeans'}
        ],
        value='agglomerative',
        style={'width': '75%', 'margin': 'auto', 'textAlign': 'center', 'marginBottom': '20px'}
    ),
 
        dcc.Graph(id='cluster-graph'),
        html.Div(id='k-slider-container', style={'display': 'none'}, children=[
        dcc.Slider(
            id='k-slider',
            min=1,
            max=10,
            step=1,
            marks={i: str(i) for i in range(1, 11)},
            value=5
        )
    ]),
        
    html.Div(id='dbscan-controls', style={'display': 'none'}, children=[
        dcc.Input(id='eps-input', type='number', placeholder='Enter eps value', value=0.5, style={'marginBottom': '10px'}),
        dcc.Input(id='min-samples-input', type='number', placeholder='Enter min_samples value', value=5, style={'marginBottom': '10px'})
    ]),
    html.Div(id='agglomerative-controls', style={'display': 'none'}, children=[
        dcc.Input(id='n-clusters-input', type='number', placeholder='Enter n_clusters value', value=5, style={'marginBottom': '10px'}),
        dcc.Dropdown(
            id='linkage-dropdown',
            options=[
                {'label': 'Ward', 'value': 'ward'},
                {'label': 'Complete', 'value': 'complete'},
                {'label': 'Average', 'value': 'average'}
            ],
            value='ward',
            style={'marginBottom': '10px'}
        ),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Euclidean', 'value': 'euclidean'},
                {'label': 'Manhattan', 'value': 'manhattan'},
                {'label': 'Cosine', 'value': 'cosine'}
            ],
            value='euclidean',
            style={'marginBottom': '20px'}
        )
    ]),
])

@app.callback(
    Output('k-slider-container', 'style'),
    [Input('algorithm-dropdown', 'value')]
)
def toggle_slider(selected_algorithm):
    if selected_algorithm == 'kmeans':
        return {'display': 'block'}
    else:
        return {'display': 'none'}
    
@app.callback(
    Output('dbscan-controls', 'style'),
    [Input('algorithm-dropdown', 'value')]
)
def toggle_dbscan_controls(selected_algorithm):
    if selected_algorithm == 'dbscan':
        return {'display': 'block'}
    else:
        return {'display': 'none'}
    
    
    
@app.callback(
    Output('agglomerative-controls', 'style'),
    [Input('algorithm-dropdown', 'value')]
)
def toggle_agglomerative_controls(selected_algorithm):
    if selected_algorithm == 'agglomerative':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('cluster-graph', 'figure'),
    [Input('algorithm-dropdown', 'value'),
     Input('k-slider', 'value'),
     Input('eps-input', 'value'),
     Input('min-samples-input', 'value'),
     Input('n-clusters-input', 'value'),
     Input('linkage-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)

def update_graph(selected_algorithm, k, eps, min_samples, n_clusters, linkage_type, metric_type):
    
    if selected_algorithm == 'agglomerative':
        fig = Agglomerative(n_clusters, linkage_type, metric_type, df)
        return fig 
    
    
    elif selected_algorithm == 'dbscan':
        fig = dbscan(eps, min_samples, df)
        return fig
    
    elif selected_algorithm == 'kmeans':
        fig = kmeans_clustering(df,k)
        return fig  


app.run_server()