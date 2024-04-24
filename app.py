import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np

from dash import html, dcc, Dash
from dash.dependencies import Input, Output, State
df_copy  = pd.read_csv('CC GENERAL.csv') 
df = pd.read_csv("prepared_data.csv")
df = df.drop(['Unnamed: 0'], axis=1)
tsne = TSNE(n_components=2, perplexity=1, learning_rate=0.1, n_iter=250)
X_tsne = tsne.fit_transform(df)
df_tsne = pd.DataFrame(data=X_tsne, columns=['x', 'y'])
input_style = {
    'margin-bottom': '10px'
}
card_style = {
    'background-color': '#f8f9fa',
    'padding': '20px',
    'margin': '20px',
    'border-radius': '10px',
    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
}
link_style = {
    'text-decoration': 'none',
    'color': '#212529',
    'cursor': 'pointer'
}

link_box_style = {
    'padding': '10px',
    'border': '1px solid #ced4da',
    'border-radius': '5px',
    'margin-bottom': '10px',
    'background-color': '#f8f9fa',
    'transition': 'background-color 0.3s'
}
def generate_percentage_bar_plot(df):
    # Calculate the percentage of each category
    counts = df['TENURE'].value_counts(normalize=True) * 100

    # Convert counts to DataFrame for plotting
    counts_df = counts.reset_index()
    counts_df.columns = ['TENURE', 'Percentage']

    # Define custom color scale
    custom_color_scale = ['#87ceeb', '#87ceeb', '#87ceeb', '#87ceeb', '#87ceeb']

    # Plot the countplot using Plotly
    fig = px.bar(counts_df, x='TENURE', y='Percentage', 
                 labels={'TENURE': 'TENURE', 'Percentage': 'Percentage'}, 
                 title='Distribution of TENURE',
                 color_discrete_sequence=custom_color_scale)  # Set custom color scale

    # Add percentage values to each bar
    for i, row in counts_df.iterrows():
        fig.add_annotation(x=row['TENURE'], y=row['Percentage'] + 2,
                           text=f"{row['Percentage']:.1f}%",
                           showarrow=False)

    return fig

def generate_histogram_like_plot(df, column_of_interest):
    # Filter the selected data and drop NaN values
    selected_data = df[[column_of_interest]].dropna()

    # Calculate the histogram
    counts, bins = np.histogram(selected_data[column_of_interest], bins=5, range=(0, 1))
    percentages = (counts / len(selected_data[column_of_interest])) * 100

    # Create the bar plot using Plotly
    fig = go.Figure()

    # Add bars for each bin
    for i in range(len(bins)-1):
        fig.add_trace(go.Bar(
            x=[(bins[i] + bins[i+1]) / 2],  # Center of the bin
            y=[percentages[i]],
            width=[(bins[i+1] - bins[i])],  # Width of the bin
            marker_color='skyblue'
        ))

        # Add annotation for each bar
        fig.add_annotation(
            x=(bins[i] + bins[i+1]) / 2,  # x-coordinate of the annotation
            y=percentages[i],  # y-coordinate of the annotation
            text=f"{percentages[i]:.1f}%",  # Text of the annotation
            showarrow=False,  # Do not show arrow
            font=dict(color='black', size=10),  # Font style of the annotation
            xshift=0,  # Horizontal shift
            yshift=5,  # Vertical shift
        )

    # Update layout
    fig.update_layout(title=f'Histogram-like plot of {column_of_interest}',
                      xaxis=dict(title=column_of_interest, range=[0, 1]),
                      yaxis=dict(title='Percentage'),
                      showlegend=False)
    
    return fig

def Agglomerative(n_clusters, linkage_type, metric_type, data):
    Agg = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage_type, metric=metric_type)
    cluster_labels = Agg.fit_predict(data)
    fig_scatter = px.scatter(df_tsne, x="x", y="y", color=cluster_labels)
    fig_scatter.update_layout(title="Agglomerative Hierarchical Clustering")
    return fig_scatter


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

BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
app = Dash(external_stylesheets=[BS], suppress_callback_exceptions=True)


# Home page layout with links to each clustering algorithm page
home_layout = html.Div([
    html.Div([
html.H1("Customer Segmentation - Credit Cards", style={'text-align': 'center'}),
    
    html.Div([
        html.Div([
            html.H2("About the Project", style={'color': '#007bff'}),
            html.P(
                "This project aims to develop a customer segmentation strategy for marketing purposes. "
                "The dataset contains information about the usage behavior of approximately 9000 active credit card holders "
                "over the last 6 months. By analyzing this data, we can identify distinct customer segments and tailor "
                "marketing strategies accordingly.",className="mb-5"
            ),
            html.Img(
                src="./assets/customer-segmentation-social.png",className='w-100 mt-5')
        ], style=card_style),
        
        html.Div([
            html.H2("Dataset Overview", style={'color': '#007bff'}),
            html.P(
                "The dataset provides details about the credit card holders, including their balances, purchase behavior, "
                "credit limits, and payment patterns. It consists of the following columns:"
            ),
            html.Ul([
                html.Li("CUST_ID: Identification of Credit Card holder (Categorical)"),
                html.Li("BALANCE: Balance amount left in their account to make purchases"),
                html.Li("BALANCE_FREQUENCY: How frequently the Balance is updated"),
                html.Li("PURCHASES: Amount of purchases made from account"),
                html.Li("ONEOFF_PURCHASES: Maximum purchase amount done in one-go"),
                html.Li("INSTALLMENTS_PURCHASES: Amount of purchase done in installment"),
                html.Li("CASH_ADVANCE: Cash in advance given by the user"),
                html.Li("PURCHASES_FREQUENCY: How frequently the Purchases are being made"),
                html.Li("ONEOFFPURCHASESFREQUENCY: How frequently Purchases are happening in one-go"),
                html.Li("PURCHASESINSTALLMENTSFREQUENCY: How frequently purchases in installments are being done"),
                html.Li("CASHADVANCEFREQUENCY: How frequently the cash in advance being paid"),
                html.Li("CASHADVANCETRX: Number of Transactions made with 'Cash in Advanced'"),
                html.Li("PURCHASES_TRX: Numbe of purchase transactions made"),
                html.Li("CREDIT_LIMIT: Limit of Credit Card for user"),
                html.Li("PAYMENTS: Amount of Payment done by user"),
                html.Li("MINIMUM_PAYMENTS: Minimum amount of payments made by user"),
                html.Li("PRCFULLPAYMENT: Percent of full payment paid by user"),
            ])
        ], style=card_style)
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'}),
    
    
    
html.Div(
    [
    html.H2("Analysis", className="mb-4", style={'text-align': 'center', 'color': '#007bff'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id='analysis-graph1', figure=generate_histogram_like_plot(df_copy, 'ONEOFF_PURCHASES_FREQUENCY'))
        ], className='col-md-6 mb-3'),
        
        html.Div([
            dcc.Graph(id='analysis-graph2',  figure=generate_histogram_like_plot(df_copy, 'PURCHASES_INSTALLMENTS_FREQUENCY'))
        ], className='col-md-6 mb-3'),
        html.Div([
            dcc.Graph(id='analysis-graph2',  figure= generate_percentage_bar_plot(df_copy))
        ], className='col-md-12 mb-3')
    
    ], className='row', style={'justify-content': 'center'}),
    

], 
style=card_style),
html.Div([
dcc.Link(
    html.Div([
        html.Div("KMeans", className="algorithm-title")
    ], className="algorithm-box"),
    href='/kmeans',
    className='link-card'
),

 dcc.Link(
    html.Div([
        html.Div("Agglomerative Hierarchical", className="algorithm-title")
    ], className="algorithm-box"),
    href='/agglomerative',
    className='link-card'
),

dcc.Link(
    html.Div([
        html.Div("DBSCAN", className="algorithm-title")
    ], className="algorithm-box"),
    href='/dbscan',
    className='link-card'
)
            ],className='pages')
    ],className='child')
],className='main')

# Page layout for KMeans
kmeans_layout = html.Div([
    html.Div([
            html.H1("KMeans Clustering", className="mb-5", style={'text-align': 'center','color': '#007bff'}),
    
    html.Div([
        html.Div([
            html.Div([
            html.Label("Enter Number Of Clusters", style={'font-weight': 'bold'},className='me-3'),
            dcc.Input(id='k-slider', type='number',value=1, placeholder='Enter number of clusters (k)', style=input_style)])
        ], className='col-md-4 d-flex flex-row justify-content-center mb-3'),
        
        html.Div([
            dcc.Graph(id='kmeans-cluster-graph')
        ], className='col-md-8')
    ], className='row', style={'justify-content': 'center'})
        
    ],style=card_style)

])


agglomerative_layout = html.Div([
    html.Div([
        html.H1("Agglomerative Hierarchical Clustering", className="mb-5", style={'text-align': 'center', 'color': '#007bff'}),
    
        html.Div([
            html.Div([
                html.Label("Enter Number Of Clusters", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.Input(id='n-clusters-input', type='number', placeholder='Enter number of clusters', value=5, style={'width': '100px'})
            ], className='col-md-4 d-flex align-items-center justify-content-center mb-3'),
        
            html.Div([
                dcc.Dropdown(
                    id='linkage-dropdown',
                    options=[
                        {'label': 'ward', 'value': 'ward'},
                        {'label': 'complete', 'value': 'complete'},
                        {'label': 'average', 'value': 'average'}
                    ],
                    value='complete',
                    style={'width': '200px'}
                ),
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[
                        {'label': 'Euclidean', 'value': 'euclidean'},
                        {'label': 'Manhattan', 'value': 'manhattan'},
                        {'label': 'Cosine', 'value': 'cosine'}
                    ],
                    value='euclidean',
                    style={'width': '200px', 'margin-left': '20px'}
                )
            ], className='col-md-8 d-flex align-items-center justify-content-center')
        ], className='row mb-3', style={'justify-content': 'center'}),
    
        html.Div([
            dcc.Graph(id='agglomerative-cluster-graph')
        ], className='row', style={'justify-content': 'center'})
    ], className='child')
], style=card_style, className='main')




# Define DBSCAN Clustering layout
dbscan_layout = html.Div([
    html.Div([
    html.H1("DBSCAN Clustering", className="mb-5", style={'text-align': 'center', 'color': '#007bff'}),
    
    html.Div([
        html.Div([
            html.Label("Enter eps", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Input(id='eps-input', type='number', placeholder='Enter eps value', value=0.5, style={'width': '100%', 'margin-bottom': '10px'})
        ], className='col-md-4 d-flex align-items-center justify-content-center mb-3'),
        
        html.Div([
            html.Label("Enter min_samples", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Input(id='min-samples-input', type='number', placeholder='Enter min_samples value', value=5, style={'width': '100%', 'margin-top': '10px'})
        ], className='col-md-4 d-flex align-items-center justify-content-center mb-3')
    ], className='row', style={'justify-content': 'center'}),
    
    html.Div([
        dcc.Graph(id='dbscan-cluster-graph', className='col-md-8', style={'margin': 'auto'})
    ], className='row', style={'justify-content': 'center'})
    ],className='child')
    
    
], style=card_style,className='main')   







app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Define the analysis page layout
analysis_page_layout = html.Div([   
    html.Div([
    html.H1("Analysis Page", className="mb-5", style={'text-align': 'center', 'color': '#007bff'}),
    
    html.Div([
        dcc.Graph(id='analysis-graph3'),
        dcc.Graph(id='analysis-graph4')
    ], className='row', style={'justify-content': 'center'}),
    
    html.Div([
        dcc.Link(html.Button("Back to Home"), href='/', className='btn btn-primary')
    ], style={'text-align': 'center', 'margin-top': '20px'})],className='child')
], style=card_style,className='main')

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/kmeans':
        return kmeans_layout
    elif pathname == '/agglomerative':
        return agglomerative_layout
    elif pathname == '/dbscan':
        return dbscan_layout
    elif pathname == '/analysis':
        return analysis_page_layout
    else:
        return home_layout  
# Callback to update KMeans clustering graph based on input value



@app.callback(
    Output('kmeans-cluster-graph', 'figure'),
    [Input('k-slider', 'value')],
    [State('url', 'pathname')]
)
def update_kmeans_graph(k, pathname):
    if pathname == '/kmeans':
        return kmeans_clustering(df, k)
    else:
        return {}

# Callback to update Agglomerative clustering graph based on input values
@app.callback(
    Output('agglomerative-cluster-graph', 'figure'),
    [Input('n-clusters-input', 'value'),
     Input('linkage-dropdown', 'value'),
     Input('metric-dropdown', 'value')],
    [State('url', 'pathname')]
)
def update_agglomerative_graph(n_clusters, linkage_type, metric_type, pathname):
    if pathname == '/agglomerative':
        return Agglomerative(n_clusters, linkage_type, metric_type, df)
    else:
        return {}

# Callback to update DBSCAN clustering graph based on input values
@app.callback(
    Output('dbscan-cluster-graph', 'figure'),
    [Input('eps-input', 'value'),
     Input('min-samples-input', 'value')],
    [State('url', 'pathname')]
)
def update_dbscan_graph(eps, min_samples, pathname):
    if pathname == '/dbscan':
        return dbscan(eps, min_samples, df)
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True)