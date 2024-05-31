import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import plotly.express as px

class RoundBlock:
    def __init__(self, engine):
        self.engine = engine
        self.rounds_df = self.load_data()
    
    def load_data(self):
        rounds_query = "SELECT * FROM rounds;"
        return pd.read_sql(rounds_query, self.engine).dropna()
    
    def create_filters(self):
        filter_columns = ['stage', 'country', 'raise']
        filters = {}
        for col in filter_columns:
            if col == 'raise':
                filters[col] = st.slider(
                    f'Диапазон {col}', 
                    self.rounds_df[col].min(), 
                    self.rounds_df[col].max(), 
                    (self.rounds_df[col].min(), self.rounds_df[col].max())
                )
            else:
                filters[col] = st.checkbox(f'{col}', True)
                if filters[col]:
                    filters[col] = st.selectbox(f'Выберите {col}', self.rounds_df[col].unique())
        return filters
    
    def apply_filters(self, filters):
        filtered_data = self.rounds_df
        for col, active in filters.items():
            if active:
                if col == 'raise':
                    filtered_data = filtered_data[(filtered_data[col] >= filters[col][0]) & (filtered_data[col] <= filters[col][1])]
                else:
                    filtered_data = filtered_data[filtered_data[col] == filters[col]]
        return filtered_data
    
    def display_dataframe(self, filtered_data):
        st.dataframe(filtered_data.drop(['id', 'key', 'project_key'], axis=1))
    
    def plot_pie_charts(self, filtered_data):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        self.plot_pie_chart(
            axes[0], 
            filtered_data, 
            'country', 
            'Страны', 
            threshold_percentage=3
        )
        
        self.plot_pie_chart(
            axes[1], 
            filtered_data, 
            'stage', 
            'Стадии', 
            threshold_percentage=3
        )
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def plot_pie_chart(self, ax, data, column, title, threshold_percentage=5):
        value_counts = data[column].value_counts().sort_values(ascending=False)
        threshold = (threshold_percentage / 100) * value_counts.sum()
        other_counts = value_counts[value_counts < threshold].sum()
        value_counts = value_counts[value_counts >= threshold]
        if other_counts >= 0.01:
            value_counts['Другие'] = other_counts
        
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        ax.set_title(title)
    
    def render(self):
        st.header("Анализ раундов инвестиций")
        
        filters = self.create_filters()
        filtered_data = self.apply_filters(filters)
        
        self.display_dataframe(filtered_data)
        
        try:
            if len(filtered_data) != 0:
                self.plot_pie_charts(filtered_data)
            else:
                st.caption('Похоже, что с такими фильтрами мы не можем построить графики - выберите другие')
        except Exception as e:
            st.caption(f'Ошибка при построении графиков: {e}')
        
