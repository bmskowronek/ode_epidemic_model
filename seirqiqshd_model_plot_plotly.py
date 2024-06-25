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

def generate_data(beta, gamma, sigma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada,
                  initial_conditions, t_points, seasonal_amplitude=2000, seasonal_period=7,
                  add_noise=False, generation=False):
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
    if generation:
        data.to_csv('simulated_data_extended.csv', index=False)
        print('saved generated data to simulated_data_extended.csv')
    return S_data, E_data, I_data, R_data, Qi_data, Qs_data, H_data, D_data



N = 38000000
initial_conditions = [N-25, 0, 25, 0, 0, 0, 0, 0]  # [S0, E0, I0, R0, Qi0, Qs0, H0, D0]

# True parameters
prob_of_infecting = 1/7
avg_no_contacts_per_individual = 25
beta_true = prob_of_infecting * avg_no_contacts_per_individual / N
sigma_true = 1/7  # Incubation period of ~7 days
gamma_true = 1/14  # Recovery rate (14 days average)
rho_true = 1/90  # Rate of loss of immunity (90 days)
eta_true = 0.1  # Hospitalization rate for infected patients (severity)
mu_true = 0.1  # Mortality rate for hospitalized patients
xi_true = 0.001  # Direct mortality rate from infection (compartment I)
alpha_true = 0.01 # Rate at which susceptible individuals enter quarantine
delta_true = 0.01  # Rate at which infected individuals enter quarantine
epsilon_true = 1/20 # Rate at which individuals leave hospital
theta_true = 1/14  # Rate at which susceptible individuals leave quarantine
lambada_true = 1/14  # Rate at which infected individuals leave quarantine


true_params = [beta_true, sigma_true, gamma_true, rho_true, eta_true, mu_true, xi_true, alpha_true, delta_true, epsilon_true, theta_true, lambada_true]

t_start, t_end, t_step = 0, 720, 1
t_points = np.arange(t_start, t_end, t_step)

S_observed, E_observed, I_observed, R_observed, Qi_observed, Qs_observed, H_observed, D_observed = generate_data(*true_params, initial_conditions, t_points)



def plotly_plot(filename = "wykres.html"):    
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
    fig.write_html(filename) 
    #if fig_show doesn't properly show the plot, uncomment the line above and view file "wykres.html"
plotly_plot(filename = "wykres1.html")
param_names = ['beta', 'gamma', 'sigma', 'rho', 'eta', 'mu', 'xi', 'alpha', 'delta', 'epsilon', 'theta', 'lambda']


beta_range = np.linspace(0.1*beta_true, beta_true, 100)
def hospital_capacity_scenario(beta_range):
    results = []
    for beta in beta_range:
        params = [beta] + true_params[1:]
        _, _, I, _, _, _, H, _ = generate_data(*params, initial_conditions, t_points)
        max_hospitalized = max(H)
        results.append((beta, max_hospitalized))
    
    capacity_results = results
    
    # Find the Beta value where capacity is reached
    capacity_threshold = 80000
    beta_at_capacity = None
    lastbeta = None  # Initialize lastbeta
    
    for beta, max_hospitalized in capacity_results:
        if max_hospitalized >= capacity_threshold:
            max_hospitalized_for_capacity = max_hospitalized
            beta_at_capacity = beta
            break
        lastbeta = beta  # Save the current beta as lastbeta before moving to the next iteration
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot([r[0] for r in capacity_results], [r[1] for r in capacity_results])
    plt.axhline(y=capacity_threshold, color='r', linestyle='--')
    plt.xlabel('Beta')
    plt.ylabel('Peak Hospitalized')
    plt.title(f'Hospital capacity scenario, alpha: {true_params[7]}, delta: {true_params[8]}')
    
    # Add subtitle with Beta value at capacity
    if beta_at_capacity:
        max_contacts = beta_at_capacity*N/prob_of_infecting
        plt.suptitle(f'Capacity of {capacity_threshold} reached at Beta ≈ {beta_at_capacity:.15f}. Max number of contacts: {max_contacts:.5f}', 
                     fontsize=10, y=0.95)
    else:
        plt.suptitle('Capacity not reached within the given Beta range', 
                     fontsize=10, y=0.95)
    
    plt.tight_layout()
    plt.show()
    if beta_at_capacity:
        return (lastbeta, max_hospitalized_for_capacity) #the one before reaching the threshold
    else:
        return None
beta_boundary = hospital_capacity_scenario(beta_range)
true_params[7] = 0.1 # alpha - Rate at which susceptible individuals enter quarantine
true_params[8] = 0.1  # delta - Rate at which infected individuals enter quarantine

hospital_capacity_scenario(beta_range)
if beta_boundary:
    S_observed, E_observed, I_observed, R_observed, Qi_observed, Qs_observed, H_observed, D_observed = generate_data(beta_boundary[0],*true_params[1:], initial_conditions, t_points)
    plotly_plot(filename = "wykres2.html")
true_params[7] = 0.1 # alpha - Rate at which susceptible individuals enter quarantine
true_params[8] = 0.8  # delta - Rate at which infected individuals enter quarantine

hospital_capacity_scenario(beta_range)
if beta_boundary:
    S_observed, E_observed, I_observed, R_observed, Qi_observed, Qs_observed, H_observed, D_observed = generate_data(beta_boundary[0],*true_params[1:], initial_conditions, t_points)
    plotly_plot(filename = "wykres3.html")


# SEASONAL MODEL
def set_parameters(prob_infecting, contacts, quarantine_rates, hospital_rates):
    N = 38000000  # Assuming this is defined elsewhere
    beta = prob_infecting * contacts / N
    sigma, gamma, rho = 1/7, 1/14, 1/90
    eta, mu, xi = 0.1, 0.1, 0.001
    alpha, delta = quarantine_rates
    epsilon, theta, lambada = hospital_rates
    return [beta, sigma, gamma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada]

def simulate_scenario(params, initial_conditions, duration):
    t_points = np.arange(0, duration, 1)
    return generate_data(*params, initial_conditions, t_points)

# Initial conditions
N = 38000000
initial_conditions = [N-25, 0, 25, 0, 0, 0, 0, 0]  # [S0, E0, I0, R0, Qi0, Qs0, H0, D0]

# Part 1: winter from december to march
params1 = set_parameters(1/2, 25, (0.01, 0.95), (1/20, 1/14, 1/14))
part1 = simulate_scenario(params1, initial_conditions, 90)

# Part 2: spring from march to june
initial_conditions2 = [comp[-1] for comp in part1]
params2 = set_parameters(1/4, 25, (0.01, 0.95), (1/20, 1/14, 1/14))
part2 = simulate_scenario(params2, initial_conditions2, 90)

# Part 3: summer from june to september
initial_conditions3 = [comp[-1] for comp in part2]
params3 = set_parameters(1/7, 25, (0.01, 0.95), (1/20, 1/14, 1/14))
part3 = simulate_scenario(params3, initial_conditions3, 90)


# Part 3: autumn from september to december
initial_conditions4 = [comp[-1] for comp in part3]
params4 = set_parameters(1/4, 25, (0.01, 0.95), (1/20, 1/14, 1/14))
part4 = simulate_scenario(params4, initial_conditions4, 90)


# Merge the data from all three parts
merged_data = []
for i in range(8):  # Assuming each part has 8 components (S, E, I, R, Qi, Qs, H, D)
    merged_component = np.concatenate((part1[i], part2[i], part3[i], part4[i]))
    merged_data.append(merged_component)

# Create a time array for the x-axis
total_time_steps = len(merged_data[0])
time_points = np.arange(total_time_steps)

# Create the Plotly figure
fig = make_subplots(rows=2, cols=2, subplot_titles=("Population Compartments", "Infected and Hospitalized", 
                                                    "Quarantined", "Deceased"))

# Plot for all compartments
compartments = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Quarantined Infected', 
                'Quarantined Susceptible', 'Hospitalized', 'Deceased']
for i, comp in enumerate(compartments):
    fig.add_trace(go.Scatter(x=time_points, y=merged_data[i], mode='lines', name=comp), row=1, col=1)

# Plot for Infected and Hospitalized
fig.add_trace(go.Scatter(x=time_points, y=merged_data[2], mode='lines', name='Infected'), row=1, col=2)
fig.add_trace(go.Scatter(x=time_points, y=merged_data[6], mode='lines', name='Hospitalized'), row=1, col=2)

# Plot for Quarantined (both Infected and Susceptible)
fig.add_trace(go.Scatter(x=time_points, y=merged_data[4], mode='lines', name='Quarantined Infected'), row=2, col=1)
fig.add_trace(go.Scatter(x=time_points, y=merged_data[5], mode='lines', name='Quarantined Susceptible'), row=2, col=1)

# Plot for Deceased
fig.add_trace(go.Scatter(x=time_points, y=merged_data[7], mode='lines', name='Deceased'), row=2, col=2)

# Add vertical lines at day 120 and day 180
for i in range(1, 3):
    for j in range(1, 3):
        fig.add_vline(x=90, line_dash="dash", line_color="red", row=i, col=j)
        fig.add_vline(x=180, line_dash="dash", line_color="red", row=i, col=j)
        fig.add_vline(x=270, line_dash="dash", line_color="red", row=i, col=j)


# Update layout
fig.update_layout(height=800, width=1200, title_text="Extended SIR Model Simulation")
fig.update_xaxes(title_text="Time (days)")
fig.update_yaxes(title_text="Number of Individuals")

# Add annotations for the three parts
fig.add_annotation(x=60, y=0, text="Spring", showarrow=False, xref="x", yref="paper", row=1, col=1)
fig.add_annotation(x=150, y=0, text="Summer", showarrow=False, xref="x", yref="paper", row=1, col=1)
fig.add_annotation(x=240, y=0, text="Autumn", showarrow=False, xref="x", yref="paper", row=1, col=1)
fig.add_annotation(x=330, y=0, text="Winter", showarrow=False, xref="x", yref="paper", row=1, col=1)
# Show the plot
fig.show()
fig.write_html('wykres_seasonal.html') 




# CUSTOM SCENARIO
# Initial conditions
N = 38000000
initial_conditions = [N-25, 0, 25, 0, 0, 0, 0, 0]  # [S0, E0, I0, R0, Qi0, Qs0, H0, D0]

# Part 1: Infected in quarantine, rest is rather normal
params1 = set_parameters(1/7, 25, (0.01, 0.99), (1/20, 1/14, 1/14))
part1 = simulate_scenario(params1, initial_conditions, 120)

# Part 2: Lockdown
initial_conditions2 = [comp[-1] for comp in part1]
params2 = set_parameters(1/7, 0.3, (1, 1), (1/20, 1/100, 1/100))
part2 = simulate_scenario(params2, initial_conditions2, 60)

# Part 3: Chaos and anti-governmental restriction defiance
initial_conditions3 = [comp[-1] for comp in part2]
params3 = set_parameters(1/7, 45, (0, 0.4), (1/10, 1/14, 1/14))
part3 = simulate_scenario(params3, initial_conditions3, 180)

# Merge the data from all three parts
merged_data = []
for i in range(8):  # Assuming each part has 8 components (S, E, I, R, Qi, Qs, H, D)
    merged_component = np.concatenate((part1[i], part2[i], part3[i]))
    merged_data.append(merged_component)

# Create a time array for the x-axis
total_time_steps = len(merged_data[0])
time_points = np.arange(total_time_steps)

# Create the Plotly figure
fig = make_subplots(rows=2, cols=2, subplot_titles=("Population Compartments", "Infected and Hospitalized", 
                                                    "Quarantined", "Deceased"))

# Plot for all compartments
compartments = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Quarantined Infected', 
                'Quarantined Susceptible', 'Hospitalized', 'Deceased']
for i, comp in enumerate(compartments):
    fig.add_trace(go.Scatter(x=time_points, y=merged_data[i], mode='lines', name=comp), row=1, col=1)

# Plot for Infected and Hospitalized
fig.add_trace(go.Scatter(x=time_points, y=merged_data[2], mode='lines', name='Infected'), row=1, col=2)
fig.add_trace(go.Scatter(x=time_points, y=merged_data[6], mode='lines', name='Hospitalized'), row=1, col=2)

# Plot for Quarantined (both Infected and Susceptible)
fig.add_trace(go.Scatter(x=time_points, y=merged_data[4], mode='lines', name='Quarantined Infected'), row=2, col=1)
fig.add_trace(go.Scatter(x=time_points, y=merged_data[5], mode='lines', name='Quarantined Susceptible'), row=2, col=1)

# Plot for Deceased
fig.add_trace(go.Scatter(x=time_points, y=merged_data[7], mode='lines', name='Deceased'), row=2, col=2)

# Add vertical lines at day 120 and day 180
for i in range(1, 3):
    for j in range(1, 3):
        fig.add_vline(x=120, line_dash="dash", line_color="red", row=i, col=j)
        fig.add_vline(x=180, line_dash="dash", line_color="red", row=i, col=j)

# Update layout
fig.update_layout(height=800, width=1200, title_text="Extended SIR Model Simulation")
fig.update_xaxes(title_text="Time (days)")
fig.update_yaxes(title_text="Number of Individuals")

# Add annotations for the three parts
fig.add_annotation(x=60, y=1, text="Inf. in quarantine", showarrow=False, xref="x", yref="paper", row=1, col=1)
fig.add_annotation(x=150, y=1, text="Lockdown", showarrow=False, xref="x", yref="paper", row=1, col=1)
fig.add_annotation(x=240, y=1, text="Chaos", showarrow=False, xref="x", yref="paper", row=1, col=1)

# Show the plot
fig.show()
fig.write_html('wykres6.html') 



#DATA GENERATION
S_observed, E_observed, I_observed, R_observed, Qi_observed, Qs_observed, H_observed, D_observed = generate_data(*true_params, initial_conditions, t_points, 
                                                                                                                 generation=True, add_noise=True, seasonal_amplitude=200000, seasonal_period=365)
generated_data = {"S":S_observed, "E":E_observed, "I":I_observed, "R":R_observed, "Qi":Qi_observed, "Qs":Qs_observed, "H":H_observed, "D":D_observed}

df = pd.DataFrame(generated_data)

initial_guess = [0.3, 1/10, 1/7, 1/90, 0.15, 0.02, 0.01, 0.1, 0.3, 1/10, 1/14, 1/14]

''' does not work/stuck in a loop
#PREDICTION USING OPTIMIZATION
def objective_function(params, t, I_observed):
    beta, sigma, gamma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada = params
    
    N = 38000000
    initial_conditions = [N-25, 0, 25, 0, 0, 0, 0, 0]
    
    # Solve the ODE system
    solution = solve_ivp(sir_model, [0, len(t)], initial_conditions, 
                         args=(beta, sigma, gamma, rho, eta, mu, xi, alpha, delta, epsilon, theta, lambada),
                         t_eval=t)
    
    # Extract the infected compartment
    I_predicted = solution.y[2]
    
    # Calculate sum of squared errors
    return np.sum((I_observed - I_predicted)**2)

# Extract the infected data from the first 180 days (winter and spring)
t = np.arange(180)
I_observed = merged_data[2][:180]

# Initial guess for parameters 
initial_guess = [0.3, 1/7, 1/14, 1/90, 0.1, 0.1, 0.001, 0.01, 0.95, 1/20, 1/14, 1/14]

# Set bounds for parameters
bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

# Perform the optimization
result = minimize(objective_function, initial_guess, args=(t, I_observed), method='L-BFGS-B', bounds=bounds)

# Extract the optimized parameters
optimized_params = result.x

# Generate predictions using the optimized parameters
solution = solve_ivp(sir_model, [0, 180], initial_conditions, args=tuple(optimized_params), t_eval=t)
I_predicted = solution.y[2]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, I_observed, label='Observed')
plt.plot(t, I_predicted, label='Predicted')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infected Individuals')
plt.title('Model Fit: Winter to Spring')
plt.legend()
plt.show()

# Calculate R-squared
ss_res = np.sum((I_observed - I_predicted)**2)
ss_tot = np.sum((I_observed - np.mean(I_observed))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared}")


# Extract the infected data from the first 180 days (infected in quarantine and lockdown)
t = np.arange(180)
I_observed = merged_data[2][:180]

# Perform the optimization (using the same objective function and initial guess as before)
result = minimize(objective_function, initial_guess, args=(t, I_observed), method='L-BFGS-B', bounds=bounds)

# Extract the optimized parameters
optimized_params = result.x

# Generate predictions using the optimized parameters
solution = solve_ivp(sir_model, [0, 180], initial_conditions, args=tuple(optimized_params), t_eval=t)
I_predicted = solution.y[2]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, I_observed, label='Observed')
plt.plot(t, I_predicted, label='Predicted')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infected Individuals')
plt.title('Model Fit: Infected in Quarantine to Lockdown')
plt.legend()
plt.show()

# Calculate R-squared
ss_res = np.sum((I_observed - I_predicted)**2)
ss_tot = np.sum((I_observed - np.mean(I_observed))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared}")
'''








