
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import socket
import random
import plotly.graph_objects as go
import pandas as pd
import  numpy as np
import plotly.express as px
from stat_func import main

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

theta =0
prob_l = [0,0]
xi_list = []
dosage = 0,
m_border =0
dosage_values = []
last_paint_dos = 0
recommend_dosage = 1

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.CERULEAN]

# Initialize DataFrame
# Initialize DataFrame
df = pd.DataFrame({
    "dosage": [0],
    "infected":  ["Yes" if i % 2 == 0 else "No" for i in range(len([0]))],
    "total_patients": np.zeros(len([0]))
})
# Plot bar plot
fig = px.bar(df, x="dosage", y="total_patients", color="infected", barmode="stack")

fig.update_layout(
    margin=dict(l=100, r=100, t=0, b=100),  # Set all margins
    xaxis=dict(type='category'),  # Set x-axis type to category
    yaxis=dict(type='linear'),  # Set y-axis type to category
    yaxis_range=[0, 1]  # Set the y-axis limit to 0 and 1
)

# Get an available port
def get_available_port():
    # Create a socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col([
                html.H3("PCR Experiment Software", className="text-center mt-4 mb-4"),
                html.Div(random.choice(LST_OF_JOKES),id="joke", className="text-center mt-4", style={"font-size": "16px", "color": "gray",}),
                html.Br(),
                html.Div(
                    dcc.Markdown('''
                        The purpose of this application is to implement an experiment using the CRM model.
                        The purpose of these experiments is to make a choice of the most correct dose at each stage in order to find the correct c values and thus actually understand what the drug with the desired MTD is according to the researchers.
                    '''),
                    id="explanation",
                    className="text-left"
                ),
                html.Hr(style={"border": "none", "border-top": "1px solid #eee", "margin": "20px 0"})
            ])
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Instructions", className="text-center"),
                        html.Div(id="prompt-text", className="text-left"),
                        dcc.Input(
                            id="user-input",
                            type="text",
                            style={"width": "100%"}  # Set the width to 100%
                        ),
                        html.Br(),
                        html.Div(
                            dbc.Button("Submit", id="submit-button", n_clicks=0, color="primary",
                                       className="text-center"),
                            style={"display": "flex", "justify-content": "center", "padding-top": "30px"}  # Center the button horizontally
                        ),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [   html.H5("EXP - properties", className="text-center"),
                        html.Div(id="user-answer", className="text-center", style={"font-size": "16px","padding-bottom":"5px"}),
                        html.Br(),
                        html.H5("Output", className="text-center"),
                        html.Div(id="output-text", className="text-center"),
                    ],
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Col([
            html.H5("Visual Output", className="text-center"),
            html.Div(
            children=[dcc.Graph(id='bar-plot', figure=fig)],
            style={"display": "flex", "justify-content": "center", "margin-top": "0px", "padding": "0px"}
        )
        ])

    ],
    className="mt-5",
)


def init_exp(n_clicks, doses_value, output_text):
    instructions_text = ""
    if n_clicks is None or n_clicks == 0:
        instructions_text = """Please write how many doses are planned in the experiment."""
    if n_clicks == 1 and doses_value is not None:
        instructions_text = """What are the xi values in the list?\n\nPlease write them in the following format \[xi_1, xi_2, ..., xi_n\]."""
    elif n_clicks == 2 and doses_value is not None:
        instructions_text = "What is the m-value of the target probability of (MTD)?"

    return instructions_text

def exp_process(n_clicks, doses_value, output_text):
    global dosage, m_border,dosage_values,last_paint_dos,df,recommend_dosage,xi_list

    if n_clicks%2==1:
        if doses_value in ["Yes","No"]:
            df.loc[(df["infected"] == doses_value) & (
                        df["dosage"] == last_paint_dos), "total_patients"] += 1
        if n_clicks>4:
            theta, recommend_dosage,prob_test  = main(df, eval(xi_list), float(m_border))

        instructions_text = f"""The recommended dose for the experiment now is {recommend_dosage}, what dose to use in the next experiment?\n\n e.g. - '1'"""
    if n_clicks % 2 == 0:
        last_paint_dos = int(doses_value)
        instructions_text = f"""Did the experimenter experience side effects? \n\nIf yes send 'Yes' if not send 'No'"""

    return instructions_text


@app.callback(
    [Output("prompt-text", "children"), Output("output-text", "children")],
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value"), State("output-text", "children")]
)
def update_prompt_text(n_clicks, doses_value, output_text):
    global  df,m_border ,prob_l,recommend_dosage,xi_list
    output_text=""
    if n_clicks<3:
        return_val = init_exp(n_clicks, doses_value, output_text)
    else:
        return_val = exp_process(n_clicks, doses_value, output_text)
        if n_clicks > 4:
            theta, recommend_dosage, prob_l = main(df, eval(xi_list), float(m_border))

            print(theta, recommend_dosage, prob_l)
            output_text = f"""The thea is : {str(round(theta,4))}\n\nThe current estimator for pro is: {str(prob_l)} """

    return dcc.Markdown(return_val),output_text

@app.callback(
    Output("user-answer", "children"),
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value"), State("user-answer", "children")]
)
def update_exp_property(n_clicks, user_input, user_answer):
    global prob_l, m_border,dosage_values,xi_list
    if n_clicks is None or n_clicks == 0:
        return "right now there is nothing here, start the processes to init the exp"  # No user answer yet
    exp_property = user_answer

    if n_clicks == 1 and user_answer is not None:
        exp_property = f"Number of doses: {user_input}"
        dosage_values = [i  for i in range(1, int(user_input) + 1) for j in range(2)]

    if n_clicks == 2 and user_answer is not None:
        exp_property += f" | Xi vector: {user_input}"
        prob_l,xi_list = user_input,user_input

    if n_clicks == 3 and user_answer is not None:
        exp_property += f" | Target m-value: {user_input}"
        m_border=user_input

    return exp_property

# Callback function to update the bar plot
@app.callback(
    Output("bar-plot", "figure"),
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value")]
)
def update_bar_plot(n_clicks, user_input):
    global dosage,prob_l, m_border,dosage_values,df,fig
    if 0<n_clicks<4:

        # Create a DataFrame
        df = pd.DataFrame({
            "dosage": dosage_values,
            "infected":  ["Yes" if i % 2 == 0 else "No" for i in range(len(dosage_values)) ],
            "total_patients": np.zeros(len(dosage_values))
        })

        # Plot bar plot
        fig = px.bar(df, x="dosage", y="total_patients", color="infected", barmode="stack")

        fig.update_layout(
            margin=dict(l=100, r=100, t=0, b=100),  # Set all margins
            xaxis=dict(type='category'),  # Set x-axis type to category
            yaxis=dict(type='linear'),  # Set y-axis type to category
            yaxis_range=[0, 1]  # Set the y-axis limit to 0 and 1
        )
    if n_clicks>4:
        temp_df = df
        fig = px.bar(temp_df, x="dosage", y="total_patients", color="infected", barmode="stack")

        fig.update_layout(
            margin=dict(l=100, r=100, t=0, b=100),  # Set all margins
            xaxis=dict(type='category'),  # Set x-axis type to category
            yaxis=dict(type='linear'),  # Set y-axis type to category
        )

    return fig




if __name__ == '__main__':
    app.run_server(debug=False,port=get_available_port())
