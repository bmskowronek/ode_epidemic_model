import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sir_model(t, y, beta, gamma, sigma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada):
    S, E, I, R, Qi, Qs, H, D = y

    
    dSdt = -beta * S * I + rho * R - alpha * S + theta * Qs
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I - eta * I - xi * I - delta * I
    dRdt = gamma * I + epsilon * H + lambada * Qi - rho * R
    dQidt = delta * I - lambada * Qi - ((eta * Qi) / 2)
    dQsdt = alpha * S - theta * Qs
    dHdt = eta * I + eta * (Qi / 2) - mu * H - epsilon * H
    dDdt = xi * I + mu * H
    
    return [dSdt, dEdt, dIdt, dRdt, dQidt, dQsdt, dHdt, dDdt]

param_names = ['β', 'σ', 'γ', 'ρ', 'η', 'μ', 'ξ', 'α', 'δ', 'ε', 'θ', 'λ']

def generate_data(beta, gamma, sigma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada, initial_conditions, t_points, add_noise=False, seasonal_amplitude=2000, seasonal_period=7):
    solution = solve_ivp(sir_model, [t_start, t_end], initial_conditions, args=(beta, gamma, sigma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada), t_eval=t_points)

    S_data, E_data, I_data, R_data, Qi_data, Qs_data, H_data, D_data = solution.y

    if add_noise:
        for data in [S_data, E_data, I_data, R_data, Qi_data, Qs_data, H_data, D_data]:
            noise = np.random.normal(0, np.max(data) * 0.05, len(data))
            data += noise
            np.maximum(data, 0, out=data)

    seasonal_variation = seasonal_amplitude * np.sin(2 * np.pi * t_points / seasonal_period)
    I_data += seasonal_variation
    I_data = np.maximum(I_data, 0)

    # Round infected values to integers
    I_data = np.round(I_data).astype(int)

    data = pd.DataFrame({
        'Time': t_points, 'Susceptible': S_data, 'Exposed': E_data, 'Infected': I_data,
        'Recovered': R_data, 'Quarantined_Infected': Qi_data, 'Quarantined_Susceptible': Qs_data,
        'Hospitalized': H_data, 'Deceased': D_data
    })

    data.to_csv('simulated_data_extended.csv', index=False)

    return S_data, E_data, I_data, R_data, Qi_data, Qs_data, H_data, D_data

def loss_function(params):
    beta, gamma, sigma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada = params
    S_data, E_data, I_data, R_data, Qi_data, Qs_data, H_data, D_data = generate_data(beta, gamma, sigma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada, initial_conditions, t_points)
    error = np.sum((I_data - I_observed) ** 2 + (H_data - H_observed) ** 2 + (D_data - D_observed) ** 2)
    return error

N = 38000000
initial_conditions = [N - 1, 0, 1, 0, 0, 0, 0, 0]  # [S0, E0, I0, R0, Qi0, Qs0, H0, D0]

# True parameters
prob_of_infecting = 1/7
avg_no_contacts_per_individual = 45
beta_true = prob_of_infecting * avg_no_contacts_per_individual / N
sigma_true = 1/7  # Incubation period of ~7 days
gamma_true = 1/14  # Recovery rate (14 days average)
rho_true = 1/60  # Rate of loss of immunity (30 days)
eta_true = 0.2  # Hospitalization rate for infected patients (severity)
mu_true = 0.05  # Mortality rate for hospitalized patients
xi_true = 0.001  # Direct mortality rate from infection (compartment I)
alpha_true = 0.6  # Rate at which susceptible individuals enter quarantine
delta_true = 0.002  # Rate at which infected individuals enter quarantine
epsilon_true = 1/14  # Rate at which individuals leave quarantine or hospital
theta_true = 1/14  # Rate at which susceptible individuals leave quarantine
lambada_true = 1/14  # Rate at which infected individuals leave quarantine

true_params = [beta_true, sigma_true, gamma_true, rho_true, eta_true, mu_true, xi_true, alpha_true, delta_true, epsilon_true, theta_true, lambada_true]

t_start, t_end, t_step = 0, 360, 1
t_points = np.arange(t_start, t_end, t_step)

S_observed, E_observed, I_observed, R_observed, Qi_observed, Qs_observed, H_observed, D_observed = generate_data(*true_params, initial_conditions, t_points)

initial_guess = [0.3, 1/10, 1/7, 1/90, 0.15, 0.02, 0.01, 0.1, 0.3, 1/10, 1/14, 1/14]


# Create the figure
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

# Add traces for each compartment with doubled line width
fig.add_trace(go.Scatter(x=t_points, y=S_observed, mode='lines', name='Susceptible', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t_points, y=E_observed, mode='lines', name='Exposed', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t_points, y=I_observed, mode='lines', name='Infected', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t_points, y=R_observed, mode='lines', name='Recovered', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t_points, y=Qi_observed, mode='lines', name='Quarantined Infected', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t_points, y=Qs_observed, mode='lines', name='Quarantined Susceptible', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t_points, y=H_observed, mode='lines', name='Hospitalized', line=dict(width=4)))
fig.add_trace(go.Scatter(x=t_points, y=D_observed, mode='lines', name='Deceased', line=dict(width=4)))

# Update layout
fig.update_layout(
    title='Extended SIR Model Parameter Observation',
    xaxis_title='Time',
    yaxis_title='Number of Individuals',
    legend_title='Compartments',
    height=900,
    width=1400
)

# Show the plot
fig.show()
#fig.write_html("wykres.html") 
#if fig_show doesn't properly show the plot, uncomment the line above and view file "wykres.html"
param_names = ['beta', 'gamma', 'sigma', 'rho', 'eta', 'mu', 'xi', 'alpha', 'delta', 'epsilon', 'theta', 'lambda']