import pandas as pd
import plotly.express as px
import numpy as np
# List of jokes - random joke
LST_OF_JOKES = ['"What made the chicken cross the road? To get to the other side of the confidence interval!"',
                '"What is the number of statisticians needed to change a light bulb? That depends. It is really a matter of power."',
                '"Why did the statistician go broke? Because he couldn\'t find the mean!"',
                '"Why do statisticians love hiking? Because it\'s where they find their confidence intervals!"',
                '"What did the statistician say when he found a new dataset? \'I\'ve got a lot of potential hypotheses!\'"',
                '"How do statisticians stay cool in the summer? They use Bayesian fans!"',
                '"Why did the statistician bring a ladder to the bar? Because he heard the drinks were on the house!"',
                '"Why did the statistician become a magician? Because he knew how to make the data disappear!"',
                '"What\'s a statistician\'s favorite type of tree? A decision tree!"',
                '"How do you catch a statistical fish? With a normal distribution net!"',
                '"Why did the statistician always carry a clock? To keep track of the \'time series\'!"',
                '"What\'s a statistician\'s favorite song? \'Regression Line\' by The Beatles!"',
                '"Why did the statistician become a chef? Because he loved cooking up random samples!"',
                '"What do statisticians do on Halloween? They go trick-or-treating for random samples!"',
                '"Why did the statistician bring a chainsaw to the data center? To cut down on outliers!"',
                '"What did one statistician say to another at the bar? \'Let\'s get some drinks and test the hypothesis of independence!\'"',
                '"Why did the statistician become a gardener? Because he loved working with p-values!"',
                '"What\'s a statistician\'s favorite board game? Probability!"',
                '"Why was the statistician always happy? Because he had a high degree of freedom!"',
                '"How do statisticians communicate with each other? By conducting a survey!"',
                '"Why did the statistician become an architect? Because he wanted to build statistical models!"',
                '"What did one statistician say to another while hiking? \'Let\'s explore the confidence interval of this mountain view!\'"'
                ]

# Global variables
xi_list = []  # A priori probabilities
theta = 0  # Estimation of the current theta
prob_l = [0, 0]  # List of current probabilities (x_i,...x_n)**theta
dosage = 0  # The number of experimental portions
m_border = 0  # MTD (Maximum Tolerated Dose)
dosage_values = []  # 2 time dosage values [1, 1, 2, 2, ..., n, n]
last_painted_dos = 0  # The last dosage that was given to the last patient
recommend_dosage = 1  # Recommended dosage to show the user
side_effects = False  # Are there any side effects? Designed to continue administering medications regardless of the probability if there are none
first_side_effects = False  # Is there a side effect for the first patient? Designed to continue with the first patient until side effects stop if so


# Initialize DataFrame
df = pd.DataFrame({
    "dosage": [0],
    "infected":  ["No" if i % 2 == 0 else "Yes" for i in range(len([0]))],
    "total_patients": np.zeros(len([0]))
})
# Plot bar plot
fig = px.bar(df, x="dosage", y="total_patients", color="infected", barmode="stack")

fig.update_layout(
    margin=dict(l=100, r=100, t=0, b=100),  # Set all margins
    xaxis=dict(title='Dosage',type='category'),  # Set x-axis type to category
    yaxis=dict(title='Patients',type='linear'),  # Set y-axis type to category
    yaxis_range=[0, 1],  # Set the y-axis limit to 0 and 1
    legend = dict(title="Side Effects")
)
