import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


class InvestorBlock:
    def __init__(self, engine):
        self.engine = engine
        self.investors_df = self.load_data()

    def load_data(self):
        investor_query = "SELECT * FROM investors;"
        return pd.read_sql(investor_query, self.engine)

    def create_filters(self):
        filter_columns = ['tier', 'country', 'retail_roi', 'investment']
        filters = {}
        for col in filter_columns:
            if col in ['retail_roi', 'investment']:
                filters[col] = st.slider(
                    f'Диапазон {col}',
                    self.investors_df[col].min(),
                    self.investors_df[col].max(),
                    (self.investors_df[col].min(), self.investors_df[col].max()),
                    key=f'slider_{col}'
                )
            else:
                filters[col] = st.checkbox(f'{col}', True, key=f'checkbox_{col}')
                if filters[col]:
                    filters[col] = st.selectbox(
                        f'Выберите {col}', self.investors_df[col].unique(), key=f'selectbox_{col}'
                    )
        return filters

    def apply_filters(self, filters):
        filtered_data = self.investors_df
        for col, value in filters.items():
            if value:
                if col in ['retail_roi', 'investment']:
                    filtered_data = filtered_data[
                        (filtered_data[col] >= value[0]) & (filtered_data[col] <= value[1])
                    ]
                else:
                    filtered_data = filtered_data[filtered_data[col] == value]
        return filtered_data

    def display_dataframe(self, filtered_data):
        st.dataframe(filtered_data.drop(
            ['id', 'category_slug', 'category_name', 'project_name', 'key', 'type', 'category_id', 'total_investments'],
            axis=1
        ))

    def plot_pie_charts(self, filtered_data):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        self.plot_pie_chart(axes[0], filtered_data, 'tier', 'Уровни', threshold_percentage=3)
        self.plot_pie_chart(axes[1], filtered_data, 'country', 'Страны', threshold_percentage=3)
        
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
        st.header("Анализ инвесторов")

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
