import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np


class GroupInvestorBlock:
    def __init__(self, engine):
        self.engine = engine
        self.rounds_by_country_df = self.load_data(self.query_rounds_by_country)
        self.rounds_by_stage_df = self.load_data(self.query_rounds_by_stage)
        self.project_categories_df = self.load_data(self.query_project_categories)

    query_rounds_by_country = """
        SELECT
            i.id AS investor_id,
            i.name AS investor_name,
            r.country,
            COUNT(DISTINCT r.id) AS num_rounds
        FROM
            investors AS i
        JOIN
            investor_round AS ir ON i.id = ir.investor_id
        JOIN
            rounds AS r ON ir.round_id = r.id
        GROUP BY
            i.id, i.name, r.country;
    """

    query_rounds_by_stage = """
        SELECT
            i.id AS investor_id,
            i.name AS investor_name,
            r.stage,
            COUNT(DISTINCT r.id) AS num_rounds
        FROM
            investors AS i
        JOIN
            investor_round AS ir ON i.id = ir.investor_id
        JOIN
            rounds AS r ON ir.round_id = r.id
        GROUP BY
            i.id, i.name, r.stage;
    """

    query_project_categories = """
        SELECT
            i.id AS investor_id,
            i.name AS investor_name,
            p.category,
            COUNT(p.category) AS category_count
        FROM
            investors AS i
        JOIN
            investor_round AS ir ON i.id = ir.investor_id
        JOIN
            rounds AS r ON ir.round_id = r.id
        JOIN
            projects AS p ON r.project_key = p.key
        GROUP BY
            i.id, i.name, p.category;
    """

    def load_data(self, query):
        return pd.read_sql(query, self.engine)

    def filter_data(self):
        name_filter = st.checkbox('Фильтровать по имени инвестора', True)
        if name_filter:
            investor_options = self.rounds_by_country_df['investor_name'].unique()
            selected_investors = st.multiselect(
                "Выберите инвесторов", investor_options, default=[investor_options[0]]
            )
            self.rounds_by_country_df = self.rounds_by_country_df[
                self.rounds_by_country_df['investor_name'].isin(selected_investors)
            ]
            self.rounds_by_stage_df = self.rounds_by_stage_df[
                self.rounds_by_stage_df['investor_name'].isin(selected_investors)
            ]
            self.project_categories_df = self.project_categories_df[
                self.project_categories_df['investor_name'].isin(selected_investors)
            ]

    def display_data(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.dataframe(self.rounds_by_country_df.drop(['investor_id'], axis=1))
        with col2:
            st.dataframe(self.rounds_by_stage_df.drop(['investor_id'], axis=1))
        with col3:
            st.dataframe(self.project_categories_df.drop(['investor_id'], axis=1))

    def combine_small_categories(self, data, threshold=5):
        total = data.sum()
        combined_data = data.copy()
        small_categories = data[data / total * 100 < threshold]
        if not small_categories.empty:
            combined_data = combined_data.drop(small_categories.index)
            combined_data['Другие'] = small_categories.sum()
        return combined_data

    def group_data(self):
        group_rounds_by_country_df = self.rounds_by_country_df.groupby('country')['num_rounds'].sum()
        group_rounds_by_stage_df = self.rounds_by_stage_df.groupby('stage')['num_rounds'].sum()
        group_project_categories_df = self.project_categories_df.groupby('category')['category_count'].sum()

        group_rounds_by_country_df = self.combine_small_categories(group_rounds_by_country_df)
        group_rounds_by_stage_df = self.combine_small_categories(group_rounds_by_stage_df)
        group_project_categories_df = self.combine_small_categories(group_project_categories_df)

        return group_rounds_by_country_df, group_rounds_by_stage_df, group_project_categories_df

    def create_annotated_pie(self, ax, data, title, labels):
        wedges, texts, autotexts = ax.pie(
            data,
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%",
            wedgeprops=dict(width=0.6),
            textprops=dict(color='black'),
            pctdistance=0.8
        )
        ax.set_title(title, fontsize=12)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            annotation_text = (
                f"{labels[i]}\n{data[i]} rounds\n({data[i] / data.sum() * 100:.1f}%)"
            )
            ax.annotate(
                annotation_text, xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                horizontalalignment=horizontalalignment, **kw
            )

            con = ConnectionPatch(
                xyA=(x, y), coordsA=ax.transData,
                xyB=(1.35 * np.sign(x), 1.4 * y), coordsB=ax.transData,
                arrowstyle="-", lw=1, color='black'
            )
            ax.add_artist(con)

    def plot_data(self):
        group_rounds_by_country_df, group_rounds_by_stage_df, group_project_categories_df = self.group_data()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        self.create_annotated_pie(
            axes[0], group_rounds_by_country_df, "Страны", group_rounds_by_country_df.index
        )
        self.create_annotated_pie(
            axes[1], group_rounds_by_stage_df, "Стадии", group_rounds_by_stage_df.index
        )
        self.create_annotated_pie(
            axes[2], group_project_categories_df, "Категории", group_project_categories_df.index
        )

        fig.tight_layout()
        st.pyplot(fig)

    def render(self):
        st.header("Анализ инвесторов группами")
        self.filter_data()
        self.display_data()
        self.plot_data()
