import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go


def plot_tsne(data_embeddings, classes, prototypes_embeddings_list):
    """

    :param data_embeddings:
    :param classes:
    :param prototypes_embeddings_list:
    :return:
    """

    # Concatenate all data
    complete_data = pd.DataFrame(data_embeddings)
    complete_data['label'] = classes
    for i, prototypes_embeddings in enumerate(prototypes_embeddings_list):
        prototypes_embeddings_df = pd.DataFrame(prototypes_embeddings)
        prototypes_embeddings_df['label'] = [f'proto_check{i+1}']*len(prototypes_embeddings_df)
        complete_data = pd.concat([complete_data, prototypes_embeddings_df])
    complete_data = complete_data.reset_index(drop=True)

    # Fit transform using TSNE
    tsne_model = TSNE(n_components=2, learning_rate='auto', init='random')
    complete_data_tsne = tsne_model.fit_transform(complete_data.drop('label', axis=1))
    complete_data_tsne = pd.DataFrame(complete_data_tsne)
    complete_data_tsne['label'] = complete_data['label']

    # Plot embeddings
    fig = go.Figure()

    for label in complete_data_tsne['label'].unique():
        # Filter by labels
        label_data_tsne = complete_data_tsne[complete_data_tsne['label'] == label].drop('label', axis=1)
        fig.add_trace(go.Scatter(
            x=label_data_tsne.iloc[:, 0], y=label_data_tsne.iloc[:, 1],
            name=label,
            mode='markers'
        ))

    # Set options common to all traces with fig.update_traces
    fig.update_traces(mode='markers', marker_line_width=0, marker_size=5)
    fig.update_layout(title='Styled Scatter',
                      yaxis_zeroline=False, xaxis_zeroline=False)
    fig.show()
