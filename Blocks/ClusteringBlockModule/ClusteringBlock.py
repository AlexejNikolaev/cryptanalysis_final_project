from sklearn.cluster import KMeans
import streamlit as st
import pydeck as pdk
import pandas as pd
import plotly.express as px


class ClusteringBlock:
    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.data = self.load_data()

    def load_data(self):
        query = """
        SELECT 
            i.id AS investor_id,
            i.name AS investor_name,
            COUNT(ir.round_id) AS total_rounds,
            SUM(r.raise) AS total_investment,
            AVG(r.raise) AS avg_investment_per_round,
            AVG(
                CASE r.stage
                    WHEN 'Seed' THEN 1
                    WHEN 'Series A' THEN 2
                    WHEN 'Series B' THEN 3
                    WHEN 'Series C' THEN 4
                    ELSE 5
                END
            ) AS avg_stage,
            i.country AS investor_country
        FROM 
            investors i
        JOIN 
            investor_round ir ON i.id = ir.investor_id
        JOIN 
            rounds r ON ir.round_id = r.id
        GROUP BY 
            i.id, i.name, i.country;
        """
        return pd.read_sql(query, self.db_engine)

    def preprocess_data(self):
        # Удаление столбцов, где все значения пустые
        self.data.dropna(axis=1, how='all', inplace=True)

        # Удаление строк, где есть хотя бы одно пустое значение
        self.data.dropna(inplace=True)

    def render(self):
        st.header("Кластеризация инвесторов")

        # Выбор количества кластеров
        n_clusters = st.selectbox("Выберите количество кластеров:", range(2, 6))

        # Кнопка для запуска кластеризации
        if st.button("Запустить кластеризацию"):
            self.preprocess_data()

            # Применение K-means кластеризации
            kmeans = KMeans(n_clusters=n_clusters)
            features = self.data[
                ['total_rounds', 'total_investment', 'avg_investment_per_round', 'avg_stage']
            ]
            self.data['cluster'] = kmeans.fit_predict(features)

            # Вывод результатов в таблице
            st.dataframe(self.data)

            # Визуализация кластеров
            fig = px.scatter(
                self.data,
                x='total_investment',
                y='avg_investment_per_round',
                color='cluster',
                hover_data=['investor_name'],
                title='Кластеры инвесторов'
            )
            st.plotly_chart(fig)
