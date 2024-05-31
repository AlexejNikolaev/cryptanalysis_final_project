import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import pydeck as pdk
from sqlalchemy import create_engine
from Blocks.MapBlockModule import MapBlock
from Blocks.ClusteringBlockModule import ClusteringBlock
from Blocks.RoundBlockModule import RoundBlock
from Blocks.InvestorBlockModule import InvestorBlock
from Blocks.PreditionBlockModule import PredictionBlock
from Blocks.GroupInvestorBlockModule import GroupInvestorBlock
from Blocks.ProjectInvestorBlockModule import ProjectInvestorBlock


class StreamlitApp:
    def __init__(self):
        self.title = "Crypto Analytic"
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)

    def render_blocks(self):
        for block in self.blocks:
            block.render()
            st.markdown("<br><br>", unsafe_allow_html=True)

    def run(self):
        st.title(self.title)
        self.render_blocks()
        
    def create_db_engine(self):
        load_dotenv()  
        db_url = (
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        return create_engine(db_url)

if __name__ == "__main__":
    app = StreamlitApp()
    engine = app.create_db_engine()

    clustering_block = ClusteringBlock(engine)
    map_block = MapBlock(engine)
    round_block = RoundBlock(engine)
    investor_block = InvestorBlock(engine)
    prediction_block = PredictionBlock(engine)
    group_investor = GroupInvestorBlock(engine)
    project_investor = ProjectInvestorBlock(engine)

    app.add_block(map_block)
    app.add_block(project_investor)
    app.add_block(round_block)
    app.add_block(investor_block)
    app.add_block(group_investor)
    app.add_block(clustering_block)
    app.add_block(prediction_block)

    app.run()
