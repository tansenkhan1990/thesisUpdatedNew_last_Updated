import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plotly_barcharts_3d(x_df, y_df, z_df, x_min=0, y_min=0, z_min='auto', step=1, color='x',
                        x_legend='auto', y_legend='auto', z_legend='auto', flat_shading=True,
                        x_title='', y_title='', z_title='', hover_info='z', *title):
    """
    Convert a dataframe in 3D barcharts similar to matplotlib ones
        Example :
        x_df = pd.Series([1, 1, 10, 10])
        y_df = pd.Series([2, 4, 2 ,4])
        z_df = pd.Series([10, 30, 20, 45])
    :param x_df: Serie or list of data corresponding to x axis
    :param y_df: Serie or list  of data corresponding to y axis
    :param z_df: Serie or list  of data corresponding to height of the bar chart
    :param x_min: Starting position for x axis
    :param y_min: Starting position for y axis
    :param z_min: Minimum value of the barchart, if set to auto minimum value is 0.8 * minimum of z_df to obtain more
    packed charts
    :param step: Distance between two barcharts
    :param color: Axis to create color, possible parameters are
    x for a different color for each change of x
    y for a different color for each change of y
    or x+y to get a different color for each bar
    :param x_legend: Legend of x axis, if set to auto the legend is based on x_df
    :param y_legend: Legend of y axis, if set to auto the legend is based on y_df
    :param z_legend: Legend of z axis, if set to auto the legend is not shown
    :param flat_shading:
    :param x_title: Title of x axis
    :param y_title: Title of y axis
    :param z_title: Title of z axis
    :param hover_info: Hover info, z by default
    :return: 3D mesh figure acting as 3D barcharts
    """
    z_df = list(pd.Series(z_df))

    if z_min == 'auto':
        z_min = 0.8 * min(z_df)

    mesh_list = []
    colors = px.colors.qualitative.Plotly
    color_value = 0

    #x_df_uniq = np.unique(x_df)
    x_df_uniq = x_df
    # y_df_uniq = np.unique(y_df)
    y_df_uniq = y_df
    len_x_df_uniq = len(x_df_uniq)
    len_y_df_uniq = len(y_df_uniq)

    z_temp_df = []

    if len(z_df) == len_x_df_uniq and len(z_df) == len_y_df_uniq:
        for x in range(len_x_df_uniq):
            for y in range(len_y_df_uniq):
                if x == y:
                    z_temp_df.append(z_df[x])
                else:
                    z_temp_df.append(None)
    z_df = z_temp_df

    for idx, x_data in enumerate(x_df_uniq):
        if color == 'x':
            color_value = colors[idx % 9]

        for idx2, y_data in enumerate(y_df_uniq):
            if color == 'x+y':
                color_value = colors[(idx + idx2 * len(y_df.unique())) % 9]

            elif color == 'y':
                color_value = colors[idx2 % 9]

            x_max = x_min + step
            y_max = y_min + step
            print('*******************')
            print(idx * len_x_df_uniq + idx2)
            # exit()

            z_max = z_df[idx * len_x_df_uniq + idx2]
            print('check *****************')
            print(z_max)
            # exit()
            if z_max is not None:

                mesh_list.append(
                    go.Mesh3d(
                        x=[x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max],
                        y=[y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min],
                        z=[z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max],
                        color=color_value,
                        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                        opacity=1,
                        flatshading=flat_shading,
                        hovertext='text',
                        hoverinfo=hover_info,
                    ))
            else:
                mesh_list.append(
                    go.Mesh3d(
                        x=[x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max],
                        y=[y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min],
                        z=[z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max],
                        color=color_value,
                        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                        opacity=0.01,
                        flatshading=flat_shading,
                        hovertext='text',
                        hoverinfo=hover_info,
                    ))

            x_min += 2 * step
        y_min += 2 * step
        x_min = 0
    y_min = 0

    if x_legend == 'auto':
        x_legend = x_df_uniq
        x_legend = [str(x_ax) for x_ax in x_legend]
    if y_legend == 'auto':
        y_legend = y_df_uniq
        y_legend = [str(y_ax) for y_ax in y_legend]
    if z_legend == 'auto':
        z_legend = None

    fig = go.Figure(mesh_list)

    fig.update_layout(scene=dict(
        xaxis=dict(
            tickmode='array',
            ticktext=x_legend,
            tickvals=np.arange(x_min, len_x_df_uniq * 2, step=2),
            title=x_title),
        yaxis=dict(
            tickmode='array',
            ticktext=y_legend,
            tickvals=np.arange(y_min, len_y_df_uniq * 2, step=2),
            title=y_title),
        zaxis=dict(title=z_title)))
    if z_legend is None:
        fig.update_layout(scene=dict(
            zaxis=dict(
                tickmode='array',
                ticktext=z_legend,
                title=z_title)),
            template="plotly_white")

    return fig


if __name__ == '__main__':
    df = pd.read_table('/home/tansen/my_files/thesisUpdatedNew/dataset/dbpediadata/result/train_original.txt',header=None)
    data = df.groupby([3, 4], ).agg({1: ['count']}).reset_index()
    data.columns = ['year', 'country', 'feature']
    data = data[
        #(
            data.year>2000
                # )
                # &
                #  ((data.country == 'Germany') | (data.country == 'Spain') |
                #   (data.country == 'Japan')|(data.country == 'United_Kingdom')
                # )
    ]
    # data = data.drop_duplicates(subset=['country'])
    data = data.drop_duplicates(subset=['year'])
    # data = data.duplicated(subset=['country']).sum()
    # print(data)
    # data['country'] = data['country'].unique()
    # data['year'] = data['year'].astype(int)
    # data.columns = ['year', 'country', 'feature']
    # features = df2[1].to_numpy(dtype=int)
    # features = features.tolist()
    features = np.squeeze(data['feature'].values)
    # features = [2, 3, 5, 10, 20]
    years = data['year'].to_numpy(dtype=int)
    years = years.tolist()
    # features = features.astype(int)
    #neighbours = ['Bd', 'De', "USA", "Uk", "UK"]
    countries = data['country'].tolist()
    #accuracies = [0.9727, 0.9994, 0.9994, 0.9995, 0.9995]
    # accuracies = np.squeeze(df2[1].values)
    #print(accuracies)
    #print(neighbours)
    #print(years)
    #exit()
    # neighbours = ['Bd', 'De', "USA", "UK",'pa','UK']
    # features = [2, 3, 5, 10,1,4]
    # accuracies = [0.9727, 0.9995, .992,2,5,7]
    plotly_barcharts_3d(years, countries, features,
                        x_title="Years", y_title="Countries", z_title="features").show()

