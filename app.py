# import
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import socket
import random
from stat_func import main
from help_var_func import *

# Get an available port
def get_available_port():
    """
    Retrieves an available port number.

    Returns:
        port (int): An available port number.

    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


external_stylesheets = [dbc.themes.CERULEAN]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = dbc.Container(
    [
                html.Title("CRM Software")  # Change the title to "CRM Software"
                # Additional head elements can be added here if needed
        ,  # Change the heading to "CRM"
        dbc.Row(
            dbc.Col([
                html.H3("CRM Experiment Software", className="text-center mt-4 mb-4"),
                html.Div(random.choice(LST_OF_JOKES), id="joke", className="text-center mt-4", style={"font-size": "16px", "color": "gray"}),
                html.Br(),
                html.Div(
                    dcc.Markdown('''By utilizing the CRM model, this application conducts experiments to determine 
                    the appropriate dosage at each stage and identify the drug's Maximum Tolerated Dose (MTD) based 
                    on desired MTD values. It uses prior probabilities for each dose and employs a maximum likelihood 
                    estimator to estimate the drug closest to the MTD at each iteration.'''),
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
                            dbc.Button("Submit", id="submit-button", n_clicks=0, color="primary", className="text-center"),
                            style={"display": "flex", "justify-content": "center", "padding-top": "30px"}  # Center the button horizontally
                        ),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [
                        html.H5("EXP - properties", className="text-center"),
                        html.Div(id="user-answer", className="text-center", style={"font-size": "16px", "padding-bottom": "5px"}),
                        html.Br(),
                        html.H5("Output", className="text-center"),
                        html.Div(id="output-text", className="text-center",style={"padding": "5px 0", "margin-bottom": "5px"})
                    ],
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Col(
            [
                html.H5("Visual Output", className="text-center"),
                html.Div(
                    children=[dcc.Graph(id='bar-plot', figure=fig)],
                    style={"display": "flex", "justify-content": "center", "margin-top": "0px", "padding": "0px"}
                )
            ]
        )

    ],
    className="mt-5",
)


@app.callback(
    Output("user-input", "value"),
    [Input("submit-button", "n_clicks")]
)
def clear_input(n_clicks):
    """
    Callback function to clear the input text when the submit button is clicked.

    Args:
        n_clicks (int): The number of times the submit button has been clicked.

    Returns:
        str: An empty string to clear the input text.

    """
    if n_clicks is not None and n_clicks > 0:
        return ""
    else:
        return None

def init_exp(n_clicks, user_input):
    """
    Initializes the experiment based on the number of clicks and user inputs.

    Args:
        n_clicks (int): The number of times the submit button has been clicked.
        doses_value (str): User input for the xi values in the experiment.
    Returns:
        str: Instructions text based on the current state of the experiment.

    """
    instructions_text = ""
    if n_clicks is None or n_clicks == 0:
        instructions_text = """Please write how many doses are planned in the experiment."""

    elif n_clicks == 1 and user_input is not None:
        instructions_text = """What are the xi values in the list?\n\nPlease write them in the following format \[xi_1, xi_2, ..., xi_n\]."""

    elif n_clicks == 2 and user_input is not None:
        instructions_text = "What is the m-value of the target probability of (MTD)?"

    return instructions_text

def check_incremental_dosage(df, last_painted_dos, recommend_dosage, dosage_values):
    """
    Check the incremental dosage in case there are no side effect patients.

    Args:
        df (pandas.DataFrame): DataFrame containing the experimental data.
        last_painted_dos (int): Last painted dosage value.
        recommend_dosage (int): Recommended dosage for the next experiment.
        dosage_values (list): List of all dosage values.

    Returns:
        int: Recommended dosage for the next experiment.
    """
    # Check if there was only one patient without side effects in the last painted dosage
    only_one_paint_last = (df.loc[(df["dosage"] == last_painted_dos) & (df["infected"] == "No"), "total_patients"] == 1).any()

    # Check if we reached the final dosage and there are still patients with side effects
    if recommend_dosage == int(len(dosage_values) / 2):
        only_one_paint_last = (df.loc[(df["dosage"] == last_painted_dos) & (df["infected"] == "No"), "total_patients"] > 0).any()

    # Determine the recommended dosage based on the conditions
    if only_one_paint_last:
        recommend_dosage = recommend_dosage + 1 if recommend_dosage < len(dosage_values) / 2 else int(len(dosage_values) / 2)

    return recommend_dosage


def process_exp_results(df, last_painted_dos, recommend_dosage, dosage_values, xi_list, m_border):
    """
    Process experimental results to determine the recommended dosage for the next experiment.

    Args:
        df (pandas.DataFrame): DataFrame containing the experimental data.
        last_painted_dos (int): Last painted dosage value.
        recommend_dosage (int): Recommended dosage for the next experiment.
        dosage_values (list): List of all dosage values doubled [1,1,2,2,..].
        xi_list (str): Xi vector used in the experiment  - a priori probabilty.
        m_border (float): Target m-value for the experiment.

    Returns:
        int: Recommended dosage for the next experiment.
    """
    global first_side_effects

    # Check if there are  patients
    positive_paints = df[df["total_patients"] > 0]

    # Check if all paints are with dosage 1 and not with side effects is Yes
    only_dosage_1 = (positive_paints["dosage"].unique() == 1).all()
    all_infected_yes = (positive_paints["infected"].unique() == "Yes").all()

    # Check if there were no  patients with side effects in the last painted dosage
    no_infected_last = (df.loc[(df["dosage"] == last_painted_dos) & (df["infected"] == "Yes"), "total_patients"] == 0).any()

    # Determine the recommended dosage based on different conditions
    if only_dosage_1 and all_infected_yes:  # Case where the first dosage always has side effects
        recommend_dosage = 1
        first_side_effects = True
        if no_infected_last:  # If the dosages without side effects continue to the next one
            recommend_dosage = check_incremental_dosage(df, last_painted_dos, recommend_dosage, dosage_values)
    else:  # All other scenarios
        theta, recommend_dosage, prob_test = main(df, eval(xi_list), float(m_border))

    return recommend_dosage


def exp_process(n_clicks, user_input, output_text):
    """
    Processes the experiment based on the number of clicks and user input.

    :param n_clicks: The number of button clicks.
    :param user_input: The user input value for doses.
    :param output_text: The output text.
    :return: The instructions text.
    """
    global dosage, m_border, dosage_values, last_painted_dos, df, recommend_dosage, xi_list, side_effects, first_side_effects

    if n_clicks % 2 == 1:
        # Increment the total patients count for the corresponding infected dosage
        df.loc[(df["infected"] == user_input) & (df["dosage"] == last_painted_dos), "total_patients"] += 1
        if user_input == "Yes":
            side_effects = True
        print(df)
        if n_clicks > 4:
            # Update the recommended dosage based on experiment results
            recommend_dosage = process_exp_results(df, last_painted_dos, recommend_dosage, dosage_values, xi_list,
                                                   m_border)

        instructions_text = f"""The recommended dose for the experiment now is {recommend_dosage}, what dose to use in the current experiment?\n\n e.g. - '1'"""
    if n_clicks % 2 == 0:
        last_painted_dos = int(user_input)
        instructions_text = f"""Did the experimenter experience side effects? \n\nIf yes send 'Yes' if not send 'No'"""

    return instructions_text


@app.callback(
    [Output("prompt-text", "children"), Output("output-text", "children")],
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value"), State("output-text", "children")]
)
def update_prompt_text(n_clicks, user_input, output_text):
    """
    Callback function to update the prompt text and output text based on user inputs.

    Args:
        n_clicks (int): The number of times the submit button has been clicked.
        user_input (str): User input for the doses value.
        output_text: The current output text.

    Returns:
        tuple: A tuple containing the updated prompt text and output text.

    """
    global df, m_border, prob_l, recommend_dosage, xi_list, first_side_effects
    output_text = ""

    if n_clicks < 3: # init of the exp - get the initial properties of the experiment (xi list, dosage number..)
        return_val = init_exp(n_clicks, user_input)
    else: # The repeated part of the experiment - request for dosage, update on side effects, request for dosage, etc.
        return_val = exp_process(n_clicks, user_input, output_text)

        if n_clicks > 4:
            # In cases without side effect or side effects all the time in the first amount, ignore the the algorithm recommendation
            if (not side_effects) or first_side_effects:
                theta, recommend_dosage_false, prob_l = main(df, eval(xi_list), float(m_border))
            else:
                theta, recommend_dosage, prob_l = main(df, eval(xi_list), float(m_border))

            output_text = f"The value of theta is: {str(round(theta, 4))}\n\n" \
                          f"The current probability estimator is: \[{str([round(prob,4) for prob in prob_l])[1:-1]}\]"
    return dcc.Markdown(return_val), dcc.Markdown(output_text)

@app.callback(
    Output("user-answer", "children"),
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value"), State("user-answer", "children")]
)
def update_exp_property(n_clicks, user_input, exp_property):
    """
    Callback function to update the experiment property displayed to the user.

    Args:
        n_clicks (int): The number of times the submit button has been clicked.
        user_input (str): User input value.
        exp_property: The current experiment property.

    Returns:
        str: The updated experiment property.

    """
    global prob_l, m_border, dosage_values, xi_list

    if n_clicks is None or n_clicks == 0:
        return "Right now there is nothing here, start the processes to initialize the experiment"  # No user answer yet

    if n_clicks == 1 and exp_property is not None:
        exp_property = f"Number of doses: {user_input}"
        dosage_values = [i for i in range(1, int(user_input) + 1) for j in range(2)]

    if n_clicks == 2 and exp_property is not None:
        exp_property += f" | Xi vector: {user_input}"
        prob_l, xi_list = user_input, user_input

    if n_clicks == 3 and exp_property is not None:
        exp_property += f" | Target m-value: {user_input}"
        m_border = user_input

    return exp_property

# Callback function to update the bar plot
@app.callback(
    Output("bar-plot", "figure"),
    [Input("submit-button", "n_clicks")],
    [State("user-input", "value")]
)
def update_bar_plot(n_clicks, user_input):
    """
    Callback function to update the bar plot figure based on user inputs.

    Args:
        n_clicks (int): The number of times the submit button has been clicked.
        user_input (str): User input value.

    Returns:
        plotly.graph_objects.Figure: The updated bar plot figure.

    """
    global dosage, prob_l, m_border, dosage_values, df, fig

    # initiate the bar plot and the df by the user input
    if 0 < n_clicks < 4:
        # Create a DataFrame
        df = pd.DataFrame({
            "dosage": dosage_values,
            "infected": ["No" if i % 2 == 0 else "Yes" for i in range(len(dosage_values))],
            "total_patients": np.zeros(len(dosage_values))
        })

        # Plot bar plot
        fig = px.bar(df, x="dosage", y="total_patients", color="infected", barmode="stack")

        fig.update_layout(
            margin=dict(l=100, r=100, t=0, b=100),  # Set all margins
            xaxis=dict(title='Dosage', type='category'),  # Set x-axis type to category
            yaxis=dict(title='Patients', type='linear'),  # Set y-axis type to category
            yaxis_range=[0, 1],  # Set the y-axis limit to 0 and 1
            legend = dict(title="Side Effects")

        )

    # Update the bar plot
    if n_clicks > 4:
        temp_df = df
        fig = px.bar(temp_df, x="dosage", y="total_patients", color="infected", barmode="stack")

        fig.update_layout(
            margin=dict(l=100, r=100, t=0, b=100),  # Set all margins
            xaxis=dict(title='Dosage', type='category'),  # Set x-axis type to category
            yaxis=dict(title='Patients', type='linear'),  # Set y-axis type to category
            legend=dict(title="Side Effects")
        )

    return fig




if __name__ == '__main__':
    app.run_server(debug=False,port=get_available_port())
    # app.run_server(debug=False)
