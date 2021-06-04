import streamlit as st
import datetime
import akshare as ak
import unicodedata
import pandas as pd
import numpy as np

list_ = ['WDMF.AX', 'VDGR.AX','UMAX.AX','ROBO.AX','QFN.AX']


def get_list():
    symbol = st.sidebar.text_input("Input Tickers")
    st.sidebar.write("You can add **US** Tickers to the portfolio")
    st.sidebar.write("eg. Input **'MCD'** for US tickers")
    if st.sidebar.button("Add Tickers"):
        list_.append(symbol)
    drop = st.sidebar.selectbox("Drop a Ticker from the current portfolio",np.sort(list_))
    if st.sidebar.button("Drop Tickers"):   
        list_.remove(drop)
    return list_

def get_date():
    today = datetime.date.today()
    start_date = st.sidebar.date_input("Selecting the Start date",datetime.date(2020,1,1))
    end_date = st.sidebar.date_input("Selecting the End date",today)
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    return start_date, end_date

def get_portf_num():
    portf_num = st.sidebar.slider("Selecting the number of simulating portfolio weights",1000,10000,5000,1000)
    return portf_num
    

def get_stock_dataframe(stock_list,return_data = "close",start='20191210',end='20201210'):
    names = [str(i) for i in stock_list]
    codes = ["sh"+str(i) if str(i)[0] == "6" else "sz"+str(i) for i in stock_list]
    data=pd.DataFrame({name:(ak.stock_zh_a_daily(symbol=code, start_date=start, end_date=end, adjust="qfq").loc[:,return_data]) for code,name in zip(codes,names)})
    return data

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
