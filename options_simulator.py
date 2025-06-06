import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Black-Scholes functions
def black_scholes_call(S, K, T, r, q, sigma):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S, K, T, r, q, sigma):
    if T <= 0: return max(K - S, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

# Greeks calculation
def delta_call(S, K, T, r, q, sigma):
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def delta_put(S, K, T, r, q, sigma):
    if T <= 0: return -1.0 if S < K else 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) - 1

def theta_call(S, K, T, r, q, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) - 
             r*K*np.exp(-r*T)*norm.cdf(d2) + 
             q*S*np.exp(-q*T)*norm.cdf(d1))
    return theta/365  # Daily theta

def theta_put(S, K, T, r, q, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T)/(2*np.sqrt(T)) + 
             r*K*np.exp(-r*T)*norm.cdf(-d2) - 
             q*S*np.exp(-q*T)*norm.cdf(-d1))
    return theta/365  # Daily theta

def create_3d_surface(K, r, q, sigma, option_type='call'):
    # Create grid
    stock_prices = np.linspace(K*0.5, K*1.5, 50)
    time_to_expiration = np.linspace(0.1, 1, 50)
    S_grid, T_grid = np.meshgrid(stock_prices, time_to_expiration)
    
    # Calculate option prices
    prices = np.zeros_like(S_grid)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            if option_type == 'call':
                prices[i,j] = black_scholes_call(S_grid[i,j], K, T_grid[i,j], r, q, sigma)
            else:
                prices[i,j] = black_scholes_put(S_grid[i,j], K, T_grid[i,j], r, q, sigma)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, T_grid, prices, cmap=cm.coolwarm)
    
    ax.set_title(f'{option_type.capitalize()} Option Price Surface')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Expiration (years)')
    ax.set_zlabel('Option Price')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Options Pricing Simulator")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    S0 = st.sidebar.slider("Current Stock Price", 50.0, 150.0, 100.0, 1.0)
    K = st.sidebar.slider("Strike Price", 50.0, 150.0, 100.0, 1.0)
    T = st.sidebar.slider("Time to Expiration (years)", 0.1, 2.0, 1.0, 0.1)
    r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 20.0, 5.0, 0.5) / 100
    q = st.sidebar.slider("Dividend Yield (%)", 0.0, 10.0, 0.0, 0.5) / 100
    sigma = st.sidebar.slider("Volatility (%)", 10.0, 80.0, 20.0, 1.0) / 100
    days_passed = st.sidebar.slider("Days Passed", 0, 364, 0, 1)
    
    remaining_T = max(T - days_passed/365, 0.001)  # Avoid division by zero
    
    # Stock price range
    S_range = np.linspace(K*0.5, K*1.5, 100)
    
    # Calculate all values
    call_prices = [black_scholes_call(S, K, remaining_T, r, q, sigma) for S in S_range]
    put_prices = [black_scholes_put(S, K, remaining_T, r, q, sigma) for S in S_range]
    
    call_intrinsic = [max(S-K, 0) for S in S_range]
    put_intrinsic = [max(K-S, 0) for S in S_range]
    
    call_time = [c - i for c, i in zip(call_prices, call_intrinsic)]
    put_time = [p - i for p, i in zip(put_prices, put_intrinsic)]
    
    call_deltas = [delta_call(S, K, remaining_T, r, q, sigma) for S in S_range]
    put_deltas = [delta_put(S, K, remaining_T, r, q, sigma) for S in S_range]
    
    call_thetas = [theta_call(S, K, remaining_T, r, q, sigma) for S in S_range]
    put_thetas = [theta_put(S, K, remaining_T, r, q, sigma) for S in S_range]
    
    # Four-panel visualization
    st.header("Four-Panel Option Analysis")
    
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Option prices
    ax1.plot(S_range, call_prices, label='Call Option', color='blue')
    ax1.plot(S_range, put_prices, label='Put Option', color='red')
    ax1.plot(S_range, call_intrinsic, '--', color='blue', alpha=0.5, label='Call Intrinsic')
    ax1.plot(S_range, put_intrinsic, '--', color='red', alpha=0.5, label='Put Intrinsic')
    ax1.axvline(x=K, color='black', linestyle='--', label='Strike Price')
    ax1.set_title(f'Option Prices (T={remaining_T:.2f} years remaining)')
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Option Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Time value decay
    ax2.plot(S_range, call_time, label='Call Time Value', color='blue')
    ax2.plot(S_range, put_time, label='Put Time Value', color='red')
    ax2.axvline(x=K, color='black', linestyle='--')
    ax2.set_title('Time Value Component')
    ax2.set_xlabel('Stock Price')
    ax2.set_ylabel('Time Value')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Delta
    ax3.plot(S_range, call_deltas, label='Call Delta', color='blue')
    ax3.plot(S_range, put_deltas, label='Put Delta', color='red')
    ax3.axvline(x=K, color='black', linestyle='--')
    ax3.set_title('Option Deltas')
    ax3.set_xlabel('Stock Price')
    ax3.set_ylabel('Delta')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Theta
    ax4.plot(S_range, call_thetas, label='Call Theta (daily)', color='blue')
    ax4.plot(S_range, put_thetas, label='Put Theta (daily)', color='red')
    ax4.axvline(x=K, color='black', linestyle='--')
    ax4.set_title('Option Thetas (Daily Time Decay)')
    ax4.set_xlabel('Stock Price')
    ax4.set_ylabel('Theta')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig1)
    
    # 3D Surface Plots
    st.header("3D Option Price Surfaces")
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(create_3d_surface(K, r, q, sigma, 'call'))
    with col2:
        st.pyplot(create_3d_surface(K, r, q, sigma, 'put'))
    
    # Current values display
    st.header("Current Option Metrics")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Call Price", f"${black_scholes_call(S0, K, remaining_T, r, q, sigma):.2f}")
    with cols[1]:
        st.metric("Put Price", f"${black_scholes_put(S0, K, remaining_T, r, q, sigma):.2f}")
    with cols[2]:
        st.metric("Call Delta", f"{delta_call(S0, K, remaining_T, r, q, sigma):.2f}")
    with cols[3]:
        st.metric("Put Delta", f"{delta_put(S0, K, remaining_T, r, q, sigma):.2f}")
    
    cols = st.columns(2)
    with cols[0]:
        st.metric("Call Theta (daily)", f"${theta_call(S0, K, remaining_T, r, q, sigma):.4f}")
    with cols[1]:
        st.metric("Put Theta (daily)", f"${theta_put(S0, K, remaining_T, r, q, sigma):.4f}")
    
    # Explanation
    st.markdown("""
    ## How to Use This Simulator
    1. Adjust parameters in the sidebar - all visualizations update in real-time
    2. **Four-Panel View** shows:
       - Option prices with intrinsic values
       - Time value decomposition
       - Delta (price sensitivity)
       - Theta (time decay)
    3. **3D Surfaces** show how prices change with both stock price and time
    4. **Current Metrics** display exact values for the current stock price
    """)

if __name__ == "__main__":
    main()