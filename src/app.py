import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from operations import DataFrameADM


import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title='DashBoard - Análise de dados', 
    layout='wide')
streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Montserrat', sans-serif;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# Set title to the page
st.title('DashBoard - Análise de dados', anchor='center')
st.markdown("""---""")


@st.cache_data
def load_data():
    dfadm = DataFrameADM()
    dfadm.prepare()
    return dfadm

@st.cache_data
def get_metrics(_dfadm):
    qtd_transacoes = _dfadm.df.shape[0]
    qtd_lojistas = _dfadm.df.lojista_id.nunique()
    return qtd_transacoes, qtd_lojistas

@st.cache_data
def filter_by_products(_dfadm, list_products):
    _dfadm.df = _dfadm.df[_dfadm.df['produto'].isin(list_products)]
    _dfadm.prepare()
    _dfadm.restore_df()
    return _dfadm

dfadm = load_data()
qtd_transacoes, qtd_lojistas = get_metrics(dfadm)

with st.container():
    dict_df_faixas = {}
    with st.sidebar.expander('Filtros'):
        n_clusters = st.selectbox(
        'Quantas faixas gostaria de analisar?',
        list(range(dfadm.min_clusters, dfadm.max_clusters + 1))
        )
        
        dado = st.selectbox(
        'Qual dado gostaria de agrupar?',
        ["valor_transacao", "receita_total_antecipacao", "receita_total", "margem_transacao"]
        )

        # produtos = st.multiselect("Quais produtos gostaria de analisar?", dfadm.df_sample['produto'].unique())
        # if produtos:
        #     filter_by_products(dfadm, produtos)



    for cluster in range(n_clusters):
        cur_df = dfadm.dict_clustering[dado]
        dict_df_faixas[f'Faixa_{cluster}'] = dfadm.df_sample[cur_df[f'{n_clusters}_kmeans']==cluster].groupby('parcela')[['valor_transacao', 'receita_total_antecipacao', 'receita_total', 'margem_transacao']].mean().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Quantidade de transações", qtd_transacoes)
        # POS 1,1
        fig = make_subplots()
        for faixa in dict_df_faixas:
            x = dict_df_faixas[faixa]['parcela'].to_numpy()
            y = dict_df_faixas[faixa]['valor_transacao'].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=f'{faixa} - Valor transacao',
                    mode='lines+markers',
                    
                )
            )
        fig.update_layout(
            title_text=f"Valor transação por parcela clusterizado por {dado}",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
            )

        )
        fig.update_xaxes(title_text="Parcelas", showgrid=False)
        # Set y-axes titles
        fig.update_yaxes(title_text="Valor receita_total", type='log', showgrid=False)
        st.plotly_chart(fig, use_container_width=True)


        # POS 2,1
        fig = make_subplots()
        for faixa in dict_df_faixas:
            x = dict_df_faixas[faixa]['parcela'].to_numpy()
            y = dict_df_faixas[faixa]['receita_total'].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=f'{faixa} - receita_total',
                    mode='lines+markers'
                )
            )
        fig.update_layout(
            title_text=f"Receita total por parcela clusterizado por {dado}",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
            )
        )
        fig.update_xaxes(title_text="Parcelas", showgrid=False)
        # Set y-axes titles
        fig.update_yaxes(title_text="Receita Total", showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Quantidade de lojistas", qtd_lojistas)
        # POS 1,2
        fig = make_subplots()
        for faixa in dict_df_faixas:
            x = dict_df_faixas[faixa]['parcela'].to_numpy()
            y = dict_df_faixas[faixa]['margem_transacao'].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=f'{faixa} - margem transacao',
                    mode='lines+markers'
                )
            )
        
        fig.update_layout(
            title_text=f"Margem na transação por parcela clusterizado por {dado}",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
            )
        )

        fig.update_xaxes(title_text="Parcelas", showgrid=False)
        # Set y-axes titles
        fig.update_yaxes(title_text="Margem %", showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # POS 2,2
        fig = make_subplots()
        for faixa in dict_df_faixas:
            x = dict_df_faixas[faixa]['parcela'].to_numpy()
            y = dict_df_faixas[faixa]['receita_total_antecipacao'].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=f'{faixa} - receita_total_antecipacao',
                    mode='lines+markers'
                )
            )
        
        fig.update_layout(
            title_text=f"Receita total antecipada por parcela clusterizado por {dado}",
            legend=dict(
                orientation="h",
                yanchor="auto",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
            )
        )

        fig.update_xaxes(title_text="Parcelas", showgrid=False)
        # Set y-axes titles
        fig.update_yaxes(title_text="Receita Total Antecipada", showgrid=False)
        st.plotly_chart(fig, use_container_width=True)







