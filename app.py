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
        dbc.Row(
            dbc.Col([
                html.H3("PCR Experiment Software", className="text-center mt-4 mb-4"),
                html.Div(random.choice(LST_OF_JOKES), id="joke", className="text-center mt-4", style={"font-size": "16px", "color": "gray"}),
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
                        html.Div(id="output-text", className="text-center"),
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


def exp_process(n_clicks, doses_value, output_text):
    global dosage, m_border,dosage_values,last_painted_dos,df,recommend_dosage,xi_list,side_effects,first_side_effects

    if n_clicks%2==1:
        df.loc[(df["infected"] == doses_value) & (
                df["dosage"] == last_painted_dos), "total_patients"] += 1
        if doses_value=="Yes":
            side_effects=True
        print(df)
        if n_clicks>4:
            positive_paints =df[df["total_patients"]>0]

            only_dosage_1 =(positive_paints["dosage"].unique()==1).all()
            all_infected_yes = (positive_paints["infected"].unique()== "Yes").all()

            no_indected_last = (df.loc[(df["dosage"] == last_painted_dos) & (df["infected"] == "Yes"), "total_patients"] == 0).any()

            if only_dosage_1 and all_infected_yes:
                recommend_dosage=1
                first_side_effects = True


            elif no_indected_last :
                only_one_paint_last = (df.loc[(df["dosage"] == last_painted_dos) & (
                            df["infected"] == "No"), "total_patients"] == 1).any()

                print("rec_dosage",recommend_dosage)
                print("len_dosage_values",int(len(dosage_values) / 2))

                if recommend_dosage == int(len(dosage_values) / 2):
                    print((df.loc[(df["dosage"] == last_painted_dos) & (
                            df["infected"] == "No"), "total_patients"] > 0).any())

                    only_one_paint_last = (df.loc[(df["dosage"] == last_painted_dos) & (
                            df["infected"] == "No"), "total_patients"] > 0).any()
                if  only_one_paint_last:
                    recommend_dosage = recommend_dosage+1 if recommend_dosage< len(dosage_values)/2 else int(len(dosage_values)/2)
            else:
                theta, recommend_dosage, prob_test = main(df, eval(xi_list), float(m_border))

        instructions_text = f"""The recommended dose for the experiment now is {recommend_dosage}, what dose to use in the next experiment?\n\n e.g. - '1'"""
    if n_clicks % 2 == 0:
        print(doses_value)
        # last_paint_dos = int(doses_value)
        last_painted_dos = int(doses_value)
        instructions_text = f"""Did the experimenter experience side effects? \n\nIf yes send 'Yes' if not send 'No'"""


    return instructions_text
#

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

            output_text = f"The theta is: {str(round(theta, 4))}\n\nThe current estimator for prob is: {str(prob_l)}"

    return dcc.Markdown(return_val), output_text

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
            "infected": ["Yes" if i % 2 == 0 else "No" for i in range(len(dosage_values))],
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

    # Update the bar plot
    if n_clicks > 4:
        temp_df = df
        fig = px.bar(temp_df, x="dosage", y="total_patients", color="infected", barmode="stack")

        fig.update_layout(
            margin=dict(l=100, r=100, t=0, b=100),  # Set all margins
            xaxis=dict(type='category'),  # Set x-axis type to category
            yaxis=dict(type='linear')  # Set y-axis type to category
        )

    return fig




if __name__ == '__main__':
    app.run_server(debug=False,port=get_available_port())
    # app.run_server(debug=False)
