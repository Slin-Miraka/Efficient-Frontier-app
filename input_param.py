import streamlit as st
import datetime

list_ = ['FB', 'TSLA', 'TWTR', 'MSFT']


def get_list():
    symbol = st.sidebar.text_input("Input Tickers")
    if st.sidebar.button("Add Tickers"):
        list_.append(symbol)
    drop = st.sidebar.selectbox("Drop a Ticker from the current portfolio",list_)
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
    

