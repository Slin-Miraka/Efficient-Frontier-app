import yfinance as yf
import numpy as np
import pandas as pd
import scipy.optimize as sco
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from input_param import get_list,get_date,get_portf_num,is_number,get_stock_dataframe

st.sidebar.header('Welcome！ o(*￣▽￣*)ブ')



author = "Miraka"
RISKY_ASSETS = get_list()
N_PORTFOLIOS =  get_portf_num()
N_DAYS = 252
RISKY_ASSETS.sort()
START_DATE,END_DATE = get_date()
n_assets = len(RISKY_ASSETS)

st.title("Efficient Frontier APP")
st.write("Edit by: ",author)
st.write("**Current components of the portfolio**",RISKY_ASSETS)

if RISKY_ASSETS == []:
    st.error('Error: Please add at least two assets to the portfolio')

t = sum([is_number(i) for i in RISKY_ASSETS])
cn_code = RISKY_ASSETS[:t]
us_code = RISKY_ASSETS[t:]







if cn_code == []:
    prices_df = yf.download(us_code, start=START_DATE,end=END_DATE, adjusted=True)
    prices_df = prices_df['Adj Close']
elif us_code == []:
    prices_df_cn = get_stock_dataframe(cn_code, start=START_DATE,end=END_DATE)
    prices_df = prices_df_cn
elif len(us_code) == 1:
    prices_df = yf.download(us_code, start=START_DATE,end=END_DATE, adjusted=True)
    prices_df = prices_df['Adj Close']
    prices_df_cn = get_stock_dataframe(cn_code, start=START_DATE,end=END_DATE)
    prices_df = prices_df.rename(us_code[0])
    prices_df = prices_df.to_frame().merge(prices_df_cn,left_on=prices_df.index,right_on=prices_df_cn.index,how= 'left')
    prices_df = prices_df.set_index("key_0")
else:
    prices_df = yf.download(us_code, start=START_DATE,end=END_DATE, adjusted=True)
    prices_df = prices_df['Adj Close']
    prices_df_cn = get_stock_dataframe(cn_code, start=START_DATE,end=END_DATE)
    prices_df = prices_df_cn.merge(prices_df,left_on=prices_df_cn.index,right_on=prices_df.index,how= 'left')
    prices_df = prices_df.set_index("key_0")

returns_df = np.log(prices_df.pct_change() + 1).dropna()



if len(RISKY_ASSETS) != len(set(RISKY_ASSETS)):
    st.error('Error: Please remove the duplicate asset from the portfolio')
elif len(RISKY_ASSETS) < 2:
    st.error('Error: The portfolio is expected to include at least two assets.')
else:
    if st.button("View stock returns over the period"):
        fig = go.Figure()
        if isinstance(returns_df, pd.Series) == True:
            fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df
                        ,name=returns_df.name
                        
                        ))
        else: 
            for idx, col_name in enumerate(returns_df):
                fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df.iloc[:,idx]
                            ,name=returns_df.columns[idx]
                            
                            ))
        fig.update_layout(height=500, width=800, title_text="Stock Returns")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Stock returns")
                
        st.plotly_chart(fig)




    if st.button("Plotting the Efficient Frontier"):
        #creat a prices dataframe for risky assets
        

        #Calculate annualized average returns and the corresponding standard deviation 
        avg_returns = returns_df.mean() * N_DAYS
        cov_mat = returns_df.cov() * N_DAYS

        #Simulate random portfolio weights:
        np.random.seed(42)
        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
        weights /= np.sum(weights, axis=1)[:, np.newaxis] #broadcast the numpy.ndarray to a higher dimension

        #Calculate the portfolio metrics:
        portf_rtns = np.dot(weights, avg_returns)
        portf_vol = []
        for i in range(0, len(weights)):
            portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))
        portf_vol = np.array(portf_vol)
        portf_sharpe_ratio = portf_rtns / portf_vol

        #Create a DataFrame containing all the data:
        portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                        'volatility': portf_vol,
                                        'sharpe_ratio': portf_sharpe_ratio})


        #2. Define functions for calculating portfolio returns and volatility:
        def get_portf_rtn(w, avg_rtns):
            return np.sum(avg_rtns * w)
        def get_portf_vol(w, avg_rtns, cov_mat):
            return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

        #3. Define the function calculating the Efficient Frontier:

        def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
            efficient_portfolios = []
            n_assets = len(avg_returns)
            args = (avg_returns, cov_mat)
            bounds = tuple((0,1) for asset in range(n_assets))
            initial_guess = n_assets * [1. / n_assets, ]
            for ret in rtns_range:
                constraints = ({'type': 'eq',
                                'fun': lambda x: get_portf_rtn(x, avg_rtns)
                                - ret},
                                {'type': 'eq',
                                'fun': lambda x: np.sum(x) - 1})
                efficient_portfolio = sco.minimize(get_portf_vol,
                                                    initial_guess,
                                                    args=args,
                                                    method='SLSQP',
                                                    constraints=constraints,
                                                    bounds=bounds)
                efficient_portfolios.append(efficient_portfolio)
            return efficient_portfolios
        ###################################################################
        min_ret_ind = np.argmin(portf_results_df.returns)
        min_ret_portf = portf_results_df.iloc[min_ret_ind,:]
        min_ret = round(min_ret_portf[0],1)

        max_ret_ind = np.argmax(portf_results_df.returns)
        max_ret_portf = portf_results_df.iloc[max_ret_ind,:]
        max_ret = round(max_ret_portf[0],1)

        max_sharp_ind = np.argmax(portf_results_df.sharpe_ratio)


        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf_rtn = portf_results_df.iloc[min_vol_ind,:]
        otm_ret = round(min_vol_portf_rtn[0],1)
        #####################################################################



        #4. Define the considered range of returns:
        rtns_range = np.linspace(min_ret, max_ret, 200)

        #5. Calculate the Efficient Frontier:
        efficient_portfolios = get_efficient_frontier(avg_returns,
                                                        cov_mat,
                                                        rtns_range)

        #6. Extract the volatilities of the efficient portfolios:
        vols_range = [x['fun'] for x in efficient_portfolios]

        ########################################################################
        ######          new range for calculate the efficient frontier    ######
        ########################################################################

        new_rtns_range = np.linspace(otm_ret, max_ret, 100)
        new_efficient_portfolios = get_efficient_frontier(avg_returns,
                                                        cov_mat,
                                                        new_rtns_range)
        new_vols_range = [x['fun'] for x in new_efficient_portfolios]

        ########################################################################
        ######          new range for calculate the efficient frontier    ######
        ########################################################################






        min_vol_ind = np.argmin(vols_range)
        min_vol_portf_rtn = rtns_range[min_vol_ind]
        min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun']
        min_vol_portf_sharp = min_vol_portf_rtn/min_vol_portf_vol

        max_sharp_ind = np.argmax((rtns_range/vols_range))
        max_sharp_portf_rtn = rtns_range[max_sharp_ind]
        max_sharp_portf_vol = efficient_portfolios[max_sharp_ind]['fun']
        max_sharp_portf_sharp = max_sharp_portf_rtn/max_sharp_portf_vol




        #Plot the Efficient Frontier:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portf_results_df.volatility, y=portf_results_df.returns
                                ,name="Simulating portfolio"
                                ,mode="markers"
                                ,opacity=0.8
                                ,marker=dict(size=5
                                            ,color = portf_results_df.sharpe_ratio
                                            ,colorscale='Viridis'
                                            #,colorbar=dict(thickness=5, tickvals=[-5, 5], ticktext=['Low', 'High'], outlinewidth=0)
                                            ,line=dict(width=1
                                                        )
                                            )
                                ))
        fig.add_trace(go.Scatter(x=vols_range, y=rtns_range
                                ,name="Edge"
                                ,mode="lines+markers"
                                ,opacity=0.9
                                ,marker=dict(size=3
                                            ,color = 1
                                            ,line=dict(width=1
                                                    ,color = 1
                                                        )
                                            )
                                ))

        fig.add_trace(go.Scatter(x=new_vols_range, y=new_rtns_range
                                ,name="Efficient Frontier"
                                ,mode="lines+markers"
                                ,opacity=1
                                ,marker=dict(size=3
                                            ,line=dict(width=2
                                                        )
                                            )
                                )) 
        fig.add_trace(go.Scatter(x=np.sqrt(returns_df.var() * N_DAYS), y=avg_returns 
                                ,name="Indivial Stock"
                                ,mode="markers+text"
                                ,textposition = "bottom center"
                                ,marker_symbol=18
                                ,text = list(avg_returns.index)
                                ,opacity=1
                                ,marker=dict(size=10
                                            ,color = "yellow"
                                            ,line=dict(width=2
                                                        )
                                            )
                                ))

        fig.add_trace(go.Scatter(x=[min_vol_portf_vol,max_sharp_portf_vol], y=[min_vol_portf_rtn,max_sharp_portf_rtn]
                                ,name="Special portfolio"
                                ,mode="markers"
                                ,marker_symbol=[204,22]
                                ,text = ["Minimum variance portfolio","Maximum sharp ratio porfolio"]
                                ,opacity=1
                                ,marker=dict(size=10
                                            ,colorscale='Viridis'
                                            ,color = "red"
                                            ,line=dict(width=1
                                                        )
                                            )
                                ))

        fig.update_layout(height=500, width=800, title_text="{} Portforlio Efficient Frontier".format(N_PORTFOLIOS))
        fig.update_xaxes(title_text="Annualize Volatility (σ)")
        fig.update_yaxes(title_text="Annualize Expected return")
                    
        st.plotly_chart(fig)
        
        
        #fig1 = make_subplots(rows=1, cols=2)#, subplot_titles=(new_df.columns[idx]+ " ex return vs. Market return",new_df.columns[idx]+"'s std residual vs. Market return"))
        #fig1.add_trace(go.Pie(labels=list(avg_returns.index), values=list(weights[np.argmax(portf_results_df.sharpe_ratio)]), name="GHG Emissions"),row=1, col=2)

        #st.plotly_chart(fig1)

        #st.write(pd.DataFrame(list(weights[np.argmax(portf_results_df.sharpe_ratio)]),columns=avg_returns.index))
        #for x, y in zip(RISKY_ASSETS, weights[np.argmax(portf_results_df.sharpe_ratio)]):
            #st.write(f'{x}: {100*y:.2f}% ', end="", flush=False)

        min_vol = pd.DataFrame([*zip(RISKY_ASSETS, np.round(efficient_portfolios[min_vol_ind]['x'],4))],columns=["Tickers","Weights"])
        min_vol = min_vol.set_index("Tickers")
        max_sharp = pd.DataFrame([*zip(RISKY_ASSETS, np.round(efficient_portfolios[max_sharp_ind]['x'],4))],columns=["Tickers","Weights"])
        max_sharp = max_sharp.set_index("Tickers")

        st.write("**Minimum Volatility portfolio -- Performance**")
        st.write("**Return:** {:.2f}%     **Volatility:** {:.2f}%     **Sharp ratio:** {:.2f}%".format(100 *min_vol_portf_rtn,100* min_vol_portf_vol,100 *min_vol_portf_sharp))

        st.write("**Minimum Volatility portfolio -- Weights**")
        st.write(min_vol.T)
        st.write("")
        st.write("")
        st.write("**Maximum Sharpe ratio portfolio -- Performance**")
        st.write("**Return:** {:.2f}%     **Volatility:** {:.2f}%     **Sharp ratio:** {:.2f}%".format(100 *max_sharp_portf_rtn,100* max_sharp_portf_vol,100 *max_sharp_portf_sharp))
        st.write("**Maximum Sharpe ratio portfolio -- Weights**")
        st.write(max_sharp.T)

