# -*- coding: utf-8 -*-

import gzip
from itertools import islice
import os
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

class MalletModel:
    """
    """
    def __init__(self, source_file):
        self.source_file = source_file
        self.params = self.extract_params
        self.model = self.extract_model

    def extract_params(self):
        """Extract the alpha and beta values from the statefile.

        Args:
            source_file (str): Path to statefile produced by Mallet.
        Returns:
            tuple: alpha (list), beta (float)
        """
        with gzip.open(self.source_file, 'r') as state:
            head = [next(state) for x in range(3)]
            # print(head)
            params = [x.decode('utf8').strip() for x in head[1:3]]
            # print(params)
        return ([float(x) for x in list(params[0].split(":")[1].split(" "))[1:]], float(params[1].split(":")[1]))

    def extract_model(self):
        """ Returns a pandas data frame where each word is a row.
        """
        return pd.read_csv(self.source_file,
                       compression='gzip',
                       sep=' ',
                       skiprows=[1,2]
                       )


def pivot_smooth_norm(df, smooth_value, rows_variable, cols_variable, values_variable):
    """
    Turns the pandas dataframe into a data matrix.
    Args:
        df (dataframe): aggregated dataframe
        smooth_value (float): value to add to the matrix to account for the priors
        rows_variable (str): name of dataframe column to use as the rows in the matrix
        cols_variable (str): name of dataframe column to use as the columns in the matrix
        values_variable(str): name of the dataframe column to use as the values in the matrix
    Returns:
        dataframe: pandas matrix that has been normalized on the rows.
    """
    matrix = df.pivot(index=rows_variable, columns=cols_variable, values=values_variable).fillna(0)
    matrix = matrix + smooth_value

    # normed = sklearn.preprocessing.normalize(matrix, norm='l1', axis=1)
    normed = matrix.div(matrix.sum(axis=1), axis=0)

    return normed

# Move from model data to a normalized frame. 
# Take into consideration which titles are included and how to organize.

def subset_by_titles(model_data, metadata, titles, alpha):
    """
    """
    dt = model_data.merge(metadata, how='left', left_on="#doc", right_on="doc_id")
    subset = dt[dt['abrev'].isin(titles)]
    grouped = subset.groupby([subset.date_formatted.dt.year, 'topic'])['token_count'].sum().reset_index(name='token_count')
    normed = pivot_smooth_norm(grouped, alpha,'date_formatted', 'topic', 'token_count')

    return normed

def subset_by_topics(model, category, labels):
    """
    """
    topic_ids = labels[labels['topic_category'].isin(category)].mallet_topic_id.tolist()
    df = model.unstack().reset_index(name="t_proportion")
    df = df[df['topic'].isin(topic_ids)]

    df = df.merge(labels, how="left", left_on='topic', right_on="mallet_topic_id")
    df["graph_label"] = df[['topic', 'topic_label']].astype(str).apply(': '.join, axis=1)

    return df

def generate_graph_data(model, category, labels):
    """
    """
    graph_set = subset_by_topics(model, category, labels)

    # Compile into form for Plotly
    data = []
    for each in graph_set['topic'].unique():
        filtered = graph_set[graph_set['topic'] == each]
        graph_obj = go.Bar(
            x = filtered['date_formatted'],
            y = filtered['t_proportion'],
            name = ''.join(filtered['graph_label'].unique())
        )
        data.append(graph_obj)
    
    return data

def topic_subset_graph_obj(data, corpus, topic):
    """
    """
    layout = go.Layout(
        legend=dict(orientation="h"),
        barmode='stack',
        height=800,
        title="Proportion of Words in {} <br>Assigned to {}, Aggregated Yearly.".format(corpus, topic)
    )

    return go.Figure(data=data, layout=layout)

def generate_visualizations(data, corpus, category_label, output_dir):
    """
    """
    iplot(topic_subset_graph_obj(data, corpus, category_label))
    plot(topic_subset_graph_obj(data, corpus, category_label), filename=os.path.join(output_dir, '{}-{}.html'.format('-'.join(category_label.split(' ')), '-'.join(corpus.split(' ')))))

# if __name__ == '__main__':
#     test = MalletModel('/Users/jeriwieringa/Dissertation/data/model_outputs/target_300_10.18497.state.gz')
#     # print(test.model()[:3])
#     print(test.params())

