import pandas as pd
import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class CountryCoord(Base):
    __tablename__ = 'country_coord'
    id = Column(Integer, primary_key=True)
    country_name = Column(String, unique=True, nullable=False)
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    
class MapBlock:
    def __init__(self, engine):
        rounds_query = "SELECT name, country FROM rounds;"
        investors_query = "SELECT name, country FROM investors;"
        self.rounds_df = pd.read_sql(rounds_query, engine)
        self.investors_df = pd.read_sql(investors_query, engine)
        self.df = pd.DataFrame()
        self.geolocator = Nominatim(user_agent="geoapiExercises")
        self.engine = engine
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.country_list = self.get_country_list()

    def get_country_list(self):
        # Читаем данные из базы данных в DataFrame
        df = pd.read_sql_table('country_coord', self.engine)
        return {row['country_name']: (row['longitude'], row['latitude']) for index, row in df.iterrows()}

    def get_country_coordinates(self, country_name):
        if pd.notna(country_name) and country_name != "NaN":
            
            if country_name=="Greece":
                location = self.country_list.get(country_name)
                print(location)
            if country_name=="Turkey":
                location = self.country_list.get(country_name)
                print(location)
            location = self.country_list.get(country_name)
            if location:
                return [location[0], location[1]]
            else:
                # Если координаты не найдены в БД, ищем через geopy и сохраняем в БД
                geopy_location = self.geolocator.geocode(country_name)
                if geopy_location:
                    longitude = geopy_location.longitude
                    latitude = geopy_location.latitude
                    # Создаем DataFrame для новой записи
                    new_country_df = pd.DataFrame([{
                        'country_name': country_name,
                        'longitude': longitude,
                        'latitude': latitude
                    }])
                    # Сохраняем новую запись в базу данных
                    new_country_df.to_sql('country_coord', self.engine, if_exists='append', index=False)
                    return [longitude, latitude]
        return None
    

    def render(self):
        points = []
        self.investors_df = self.investors_df.dropna()
        self.rounds_df = self.rounds_df.dropna()
        st.header("Карта инвесторов и раундов")

        # Добавляем фильтр
        filter_option = st.selectbox("Выберите данные для отображения", ["Все", "Инвесторы", "Раунды"])

        country_investors = {}
        country_rounds = {}

        if filter_option in ["Все", "Инвесторы"]:
            for _, row in self.investors_df.iterrows():
                country = row['country']
                if country == "NaN":
                    continue
                investment = row['name']
                if country in country_investors:
                    country_investors[country].append(investment)
                else:
                    country_investors[country] = [investment]

        if filter_option in ["Все", "Раунды"]:
            for _, row in self.rounds_df.iterrows():
                country = row['country']
                if country == "NaN":
                    continue
                round_name = row['name']
                if country in country_rounds:
                    country_rounds[country].append(round_name)
                else:
                    country_rounds[country] = [round_name]

        for country in set(country_investors.keys()).union(set(country_rounds.keys())):
            investors = country_investors.get(country, [])
            rounds = country_rounds.get(country, [])
            count = len(investors) + len(rounds)
            text_parts = []
            if investors:
                text_parts.append(f"Investors: {', '.join(investors)}")
            if rounds:
                text_parts.append(f"Rounds: {', '.join(rounds)}")
            text = "; ".join(text_parts)
            coordinates = self.get_country_coordinates(country)
            if coordinates:
                points.append({
                    "country": country,
                    "count": count,
                    "text_info": text,
                    "coordinates": coordinates
                })

        self.df = pd.DataFrame(points)

        view_state = pdk.ViewState(
            latitude=20.0,
            longitude=0.0,
            zoom=1,
            pitch=50  # Увеличиваем pitch для 3D эффекта
        )

        # Функция для определения цвета в зависимости от count
        def get_color(count):
            if count < 5:
                return [255, 0, 0, 128]  # Полупрозрачный красный
            elif count < 10:
                return [255, 165, 0, 128]  # Полупрозрачный оранжевый
            else:
                return [0, 128, 0, 128]  # Полупрозрачный зеленый

        # Добавляем цвет в DataFrame
        self.df['color'] = self.df['count'].apply(get_color)

        layer = pdk.Layer(
            'ScatterplotLayer',
            data=self.df,
            get_position='coordinates',
            get_radius='100000',
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
        )

        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "Общее кол-во: {count}\n{text_info}"}
        )

        st.pydeck_chart(deck)
