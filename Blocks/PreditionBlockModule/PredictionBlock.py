import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class PredictionBlock:
    def __init__(self, engine):
        self.engine = engine
        self.data = self.load_data()

    def load_data(self):
        rounds_query = """
        SELECT date_trunc('month', date) AS month, COUNT(*) AS rounds
        FROM rounds
        GROUP BY 1
        ORDER BY 1;
        """
        df = pd.read_sql(rounds_query, self.engine)
        df['month'] = pd.to_datetime(df['month'])
        
        # Определение выбросов с использованием IQR
        Q1 = df['rounds'].quantile(0.25)
        Q3 = df['rounds'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Фильтрация выбросов
        df = df[(df['rounds'] >= lower_bound) & (df['rounds'] <= upper_bound)]
        return df

    def render(self):
        st.header("Прогнозирование кол-ва раундов по месяцам")

        # Определение доступного числа месяцев
        available_months = len(self.data) - 3

        # Слайдер для выбора количества месяцев для анализа
        months_for_analysis = st.slider(
            "Кол-во месяцев на основании, который происходит прогнозирование",
            min_value=1, max_value=available_months, value=12)

        # Слайдер для выбора количества месяцев для прогнозирования
        months_for_forecast = st.slider(
            "Кол-во месяцев в будущем для которых строиться прогноз",
            min_value=1, max_value=12, value=3)

        # Чекбоксы для выбора методов прогнозирования
        methods = []
        if st.checkbox("Случайный лес"):
            methods.append("Случайный лес")
        if st.checkbox("Градиентный бустинг"):
            methods.append("Градиентный бустинг")
        if st.checkbox("Линейная регрессия"):
            methods.append("Линейная регрессия")

        if not methods:
            st.warning("Пожалуйста, выберите хотя бы один метод прогнозирования.")
            return

        # Проверка на выбор всех необходимых параметров
        if months_for_analysis and months_for_forecast and methods:
            forecast_dfs, mses, r2s = self.predict(months_for_analysis, months_for_forecast, methods)

            # Отображение результатов в формате 2x2
            st.subheader("Результаты прогнозирования")
            cols = st.columns(2)
            for i, (method, forecast_df) in enumerate(forecast_dfs.items()):
                with cols[i % 2]:
                    st.write(f"Метод: {method}")

                    fig, ax = plt.subplots(figsize=(10, 5))
                    chart_data = forecast_df.set_index('Month')[['Actual', 'Forecast']].clip(lower=0)  # Убираем отрицательные значения

                    # Добавление третьей с конца точки реальных данных к прогнозируемым значениям
                    third_last_actual = chart_data['Actual'].iloc[-3]
                    forecast_with_start = pd.concat([pd.Series([third_last_actual], index=[chart_data.index[-3]]), chart_data['Forecast']])

                    ax.plot(chart_data.index, chart_data['Actual'], label='Actual', color='blue')
                    ax.plot(forecast_with_start.index, forecast_with_start.values, label='Forecast', color='green', linestyle='--')
                    ax.vlines(forecast_with_start.index, 0, forecast_with_start.values, colors='red', linestyles='dotted', linewidth=0.5)
                    ax.set_title(f"Метод: {method}\nMSE: {mses[method]:.2f} R²: {r2s[method]:.2f}")
                    ax.set_xlabel("Месяц")
                    ax.set_ylabel("Количество раундов")
                    ax.legend()
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_minor_locator(mdates.MonthLocator())
                    fig.autofmt_xdate()
                    st.pyplot(fig)

                    st.write(forecast_df)

    def predict(self, months_for_analysis, months_for_forecast, methods):
        # Заполнение отсутствующих месяцев нулями
        self.data.set_index('month', inplace=True)
        self.data = self.data.asfreq('MS').fillna(0).reset_index()

        # Обрезка последних 2 месяцев для дополнительного прогнозирования
        train_data = self.data.iloc[:-2]
        test_data = self.data.iloc[-2:]

        # Подготовка обучающих данных
        X_train = np.arange(len(train_data) - months_for_analysis, len(train_data)).reshape(-1, 1)
        y_train = train_data['rounds'][-months_for_analysis:]

        # Инициализация словарей для хранения прогнозов и метрик
        forecasts = {}
        mses = {}
        r2s = {}

        for method in methods:
            # Выбор модели
            if method == "Случайный лес":
                model = RandomForestRegressor()
            elif method == "Градиентный бустинг":
                model = GradientBoostingRegressor()
            else:
                model = LinearRegression()

            # Обучение модели
            model.fit(X_train, y_train)

            # Прогнозирование последних 2 месяцев необрезанной даты
            X_test = np.arange(len(train_data), len(train_data) + 2).reshape(-1, 1)
            y_test_pred = model.predict(X_test)

            # Вычисление метрик
            mse = mean_squared_error(test_data['rounds'], y_test_pred)
            r2 = r2_score(test_data['rounds'], y_test_pred)
            mses[method] = mse
            r2s[method] = r2

            # Подготовка данных для прогнозирования указанных месяцев
            pred_data = self.data.copy()
            pred_data['Forecast'] = np.nan
            pred_data.loc[len(train_data):len(train_data) + 1, 'Forecast'] = y_test_pred

            # Прогнозирование указанных месяцев
            X_future = np.arange(len(train_data) + 2, len(train_data) + 2 + months_for_forecast).reshape(-1, 1)
            y_future_pred = model.predict(X_future)
            future_dates = pd.date_range(start=pred_data['month'].iloc[-1] + pd.DateOffset(months=1), periods=months_for_forecast, freq='MS')
            future_df = pd.DataFrame({'month': future_dates, 'rounds': np.nan, 'Forecast': y_future_pred})

            # Объединение данных
            pred_data = pd.concat([pred_data, future_df], ignore_index=True)

            # Переименование столбца для графика
            pred_data.rename(columns={'month': 'Month', 'rounds': 'Actual'}, inplace=True)

            forecasts[method] = pred_data

        return forecasts, mses, r2s

