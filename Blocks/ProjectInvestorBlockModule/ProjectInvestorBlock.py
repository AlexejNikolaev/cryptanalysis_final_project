import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.colors as mcolors
import networkx as nx
import plotly.graph_objs as go

class ProjectInvestorBlock:
    def __init__(self, engine):
        self.engine = engine
        self.investors_project_df = self.load_data()

    def load_data(self):
        query_investors_project = """
            SELECT 
                i.name AS investor_name,
                p.name AS project_name,
                r.name AS round_name,
                r.date AS round_date,
                r.raise AS round_raise,
                r.stage AS round_stage,
                r.country AS round_country
            FROM 
                investors i
            JOIN 
                investor_round ir ON i.id = ir.investor_id
            JOIN 
                rounds r ON ir.round_id = r.id
            JOIN 
                projects p ON r.project_key = p.key
            ORDER BY 
                i.name, p.name, r.date;
        """
        return pd.read_sql(query_investors_project, self.engine)

    def create_filters(self):
        unique_countries = self.investors_project_df['round_country'].unique()
        unique_stages = self.investors_project_df['round_stage'].unique()

        selected_countries = st.multiselect('Выберите страны', unique_countries, default=unique_countries[0])
        selected_stages = st.multiselect('Выберите стадии', unique_stages, default=unique_stages[:3])

        return selected_countries, selected_stages

    def filter_data(self, selected_countries, selected_stages):
        filtered_df = self.investors_project_df[
            (self.investors_project_df['round_country'].isin(selected_countries)) &
            (self.investors_project_df['round_stage'].isin(selected_stages))
        ]
        return filtered_df

    def display_dataframe(self, filtered_df):
        st.dataframe(filtered_df)

    def plot_graph(self, filtered_df):
        G = nx.Graph()
        unique_stages = self.investors_project_df['round_stage'].unique()
        colors = list(mcolors.TABLEAU_COLORS.values())
        stage_colors = {stage: colors[i % len(colors)] for i, stage in enumerate(unique_stages)}

        for _, row in filtered_df.iterrows():
            investor = row['investor_name']
            project = row['project_name']
            stage = row['round_stage']
            G.add_node(investor, type='investor')
            G.add_node(project, type='project')
            G.add_edge(investor, project, stage=stage)

        pos = nx.spring_layout(G, dim=3)  # 3D layout

        # Extract 3D positions
        x_nodes = [pos[node][0] for node in G.nodes()]
        y_nodes = [pos[node][1] for node in G.nodes()]
        z_nodes = [pos[node][2] for node in G.nodes()]

        edge_trace = []
        for stage, color in stage_colors.items():
            edge_x = []
            edge_y = []
            edge_z = []
            for u, v in G.edges():
                if G[u][v]['stage'] == stage:
                    edge_x.extend([pos[u][0], pos[v][0], None])
                    edge_y.extend([pos[u][1], pos[v][1], None])
                    edge_z.extend([pos[u][2], pos[v][2], None])
            edge_trace.append(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color=color, width=2), name=stage))

        node_colors = ['lightblue' if G.nodes[node]['type'] == 'investor' else 'lightgreen' for node in G]
        node_trace = go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers+text',
            marker=dict(size=10, color=node_colors, opacity=0.8),
            text=[node for node in G.nodes()],
            textposition='top center'
        )

        layout = go.Layout(
            title='3D Graph of Investors and Projects',
            showlegend=True,
            legend=dict(x=0, y=1),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest'
        )

        fig = go.Figure(data=edge_trace + [node_trace], layout=layout)
        st.plotly_chart(fig)

    def render(self):
        selected_countries, selected_stages = self.create_filters()
        filtered_df = self.filter_data(selected_countries, selected_stages)
        self.display_dataframe(filtered_df)
        try:
            if not filtered_df.empty:
                self.plot_graph(filtered_df)
            else:
                st.header('Упс! Что-то пошло не так - попробуйте выбрать другие фильтры.')
        except Exception as e:
            st.header(f'Ошибка: {e}')

