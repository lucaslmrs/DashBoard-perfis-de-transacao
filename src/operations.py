import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
import random
from time import time

from clustering.clustering import KMeansRunner, KMeansConstrainedRunner

import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


class DataFrameADM:
    def __init__(self) -> None:
        self.data_path = os.getenv("DATA_PATH")
        self.qtd_lojistas = int(os.getenv("CLUSTER_SAMPLE_SIZE_OF_LOJISTAS"))
        self.qtd_transacoes = int(os.getenv("CLUSTER_SAMPLE_SIZE"))
        self.min_clusters = int(os.getenv("MIN_CLUSTERS"))
        self.max_clusters = int(os.getenv("MAX_CLUSTERS"))
                
        self.kmeans_runner = KMeansRunner()
        self.dict_clustering = {}
        self.dict_scores = {}
        self.load_data()

    def load_data(self):
        p = 0.01
        # df = pd.read_csv(self.data_path, header=0, skiprows=lambda i: i>0 and random.random() > p)
        df = pd.read_csv(self.data_path)
        self.__df_original = df[df['status_transacao']=='aprovado']
        self.restore_df()
        return self.df

    def restore_df(self):
        self.df = self.__df_original.copy()
        return self.df

    def _get_sample_by_lojistas(self, df):
        lojistas = df['lojista_id'].unique()
        sample_lojistas = np.random.choice(lojistas, self.qtd_lojistas, replace=False)
        self.df_sample_by_lojistas = df[df['lojista_id'].isin(sample_lojistas)]
        return self.df_sample_by_lojistas

    def _get_random_sample(self, df):
        if self.qtd_transacoes > len(df):
            return self.df_sample
        self.df_sample = df.sample(n=self.qtd_transacoes, replace=False, random_state=100)
        return self.df_sample

    def load_and_update_clustering(self, column):
        df_clustering = self.df_sample[[column]]
        self.dict_clustering[column], self.dict_scores[column] = self.kmeans_runner.apply_kmeans(df_clustering, range_list=range( self.min_clusters, self.max_clusters + 1))
        return self.dict_clustering[column], self.dict_scores[column]

    def prepare(self):
        self._get_random_sample(self.df)
        self.df_sample['margem_transacao'] = round(self.df_sample['receita_total'] / self.df_sample['valor_transacao'] * 100, 2)

        self.load_and_update_clustering('valor_transacao')
        self.load_and_update_clustering('receita_total')
        self.load_and_update_clustering('receita_total_antecipacao')
        self.load_and_update_clustering('margem_transacao')
