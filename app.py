# Include these if using matplotlib
# import matplotlib
# matplotlib.use('Agg') # Use 'Agg' backend for rendering to file

from flask import Flask, render_template, request, session, flash, jsonify
import numpy as np
import os
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json

np.set_printoptions(precision=8, suppress=False)


app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed to encrypt session cookies

##############################
###### FUNCTIONS #############
##############################

def generateData(n):
    data = np.random.uniform(-10,10,(n,2))
    zeros_column = np.zeros((data.shape[0], 1))
    data_with_zeros = np.hstack((data, zeros_column))
    return data_with_zeros

def findRandom(data,k):
    rand_idx = np.random.choice(data.shape[0], size=k, replace=False)
    k_centers = data[rand_idx]
    return k_centers

def findFarthest(data, k):
    n_points = data.shape[0]
    centers = []
    first_center_idx = np.random.choice(n_points)
    centers.append(data[first_center_idx])

    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - center) for center in centers) for x in data])
        next_center_idx = np.argmax(distances)
        centers.append(data[next_center_idx])
    k_centers = np.copy(centers)
    return k_centers

def findKMeansPlusPlus(data,k):
    n_points = data.shape[0]
    centers = []
    first_center_idx = np.random.choice(n_points)
    centers.append(data[first_center_idx])

    for _ in range(1,k):
        squared_dist = np.array([min(np.linalg.norm(x - center)**2 for center in centers) for x in data])
        probs = squared_dist / squared_dist.sum()
        next_center_idx = np.random.choice(n_points, p=probs)
        centers.append(data[next_center_idx])
    k_centers = np.copy(centers)
    return k_centers

# def process_click():
#     clicked_point = request.json.get('clicked_point', {})
#     print("Clicked Point: ", clicked_point)
#     return jsonify({'status': 'success', 'point_received': clicked_point})

def findClosest(data,k_centers):
    for i in range(len(data)):
        dist = []
        datax = data[i][0]
        datay = data[i][1]
        for j in range(len(k_centers)):
            #1 refers to current point, 2 refers to a k_center point
            # distance = sqrt(sq(x2-x1) + sq(y2-y1))
            distance = np.sqrt(np.square(k_centers[j][0] - datax) + np.square(k_centers[j][1] - datay))
            dist.append(distance)
            
        
        # find min value/index in dist. That refers to the closest point
        #add index of closest k_centers point to the 3rd column of data
        data[i][2] = np.argmin(dist)
    return data

def computeCenters(data,k_centers):
    identifiers = data[:,2]
    unique_ids = np.unique(identifiers)
    for i, uid in enumerate(unique_ids):
        mask = (identifiers == uid)
       # newdata = data[mask]
       # newdata[:,1]
        k_centers[i,0] = np.mean(data[mask][:,0])
        k_centers[i,1] = np.mean(data[mask][:,1])
    return k_centers


def plot(data):
    x_values = data[:,0]
    y_values = data[:,1]

    # Create Plotly plot
    #fig = px.scatter(x=x_values, y=y_values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(size=8),
    ))

    fig.update_layout(
            title='Scatter Plot by Identifiers',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            # xaxis=dict(range=[-15, 15]),
            # yaxis=dict(range=[-15, 15]),
            showlegend=True,
            legend=dict(title="Legend"),
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            autosize=False,
            width=600,
            height=600,
        )
    # Convert Plotly plot to JSON and send to the template
    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig
    # graphJSON = json.dumps(fig, cls=px.utils.PlotlyJSONEncoder)
    # return graphJSON

def plotDataKMeans(data,k_centers):
    x_values = data[:,0]
    y_values = data[:,1]
    identifiers = data[:,2]
    unique_ids = np.unique(identifiers)
    colors = px.colors.qualitative.Plotly[:len(unique_ids)]
    fig = go.Figure()
    print("Ok so far...")

    for i, uid in enumerate(unique_ids):
        mask = (identifiers == uid)
        fig.add_trace(go.Scatter(
            x=x_values[mask],
            y=y_values[mask],
            mode='markers',
            marker=dict(color=colors[i], size=8),
            name=f'ID {int(uid)}'
        ))
    fig.add_trace(go.Scatter(
        x=k_centers[:,0],
        y=k_centers[:,1],
        mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name='Centroid'
        ))
    fig.update_layout(
        title='Scatter Plot by Identifiers',
        xaxis_title='X Position',
        yaxis_title='Y Position',
        # xaxis=dict(range=[-15, 15]),
        # yaxis=dict(range=[-15, 15]),
        showlegend=True,
        legend=dict(title="Legend"),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        autosize=False,
        width=600,
        height=600,
    )
    return fig

    # Create Plotly plot
    # fig = px.scatter(x=x_values, y=y_values)
 


##############################
######### Global Variables ###
##############################
# n = 100


##############################
####### Flask Logic ##########
##############################

@app.route('/', methods=['GET', 'POST'])
def index():
    # Define Initial Variables
    graphJSON = None
    number = 100
    k_value = 3
    selected_method = 'random' #default selection
    clicked_points = []

    if request.method == 'POST':
        # clicked_point = request.json.get('clicked_point', {})
        # selected_points = session.get('selected_points', [])
        # selected_points.append(clicked_point)
        # session['selected_points'] = selected_points
        # print("Selected Points: ", selected_points)


        try:
            number = request.form.get('n_points', None)
            k_value = request.form.get('k', None)
            selected_method = request.form.get('init_method', 'random')
            if number is None or number == '':
                raise ValueError("Number input cannot be empty.")
            if k_value is None or k_value == '':
                raise ValueError("k value input cannot be empty.")
            n = int(number)
            k = int(k_value)
            action = request.form['action']
            init_method = request.form['init_method']

            selected_points = None
            if init_method == 'manual' and 'selected_points' in request.form:
                selected_points_str = request.form['selected_points']
                if selected_points_str:
                    try:
                        selected_points = json.loads(request.form['selected_points'])
                        print("Selected points: ", selected_points)
                        points_list = [[point['x'], point['y']] for point in selected_points]
                        z_column = np.zeros((len(points_list), 1))
                        pts_with_zeros = np.hstack((points_list, z_column))
                        k_centers = np.copy(pts_with_zeros)
                        session['k_centers'] = k_centers


                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        selected_points = None

            print("\nSTARTING REQUEST FROM FORM.....")
            print("n = ", n)
            print("k = ", k)
            print("action = ", action)
            print("init method = ", init_method)

        
            if action == 'generate':
                # try:
                #     n = session['n']
                # finally:
                #     data = generateData(n)
                
                data = generateData(n)
                fig = plot(data)


                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                session['data'] = data.tolist()
                session['n'] = n
                session['k'] = k
                try:
                    session.pop('k_centers', None)
                finally:
                    print("Could not clear k_centers")

            if action == 'step':
                # Step should be greyed out if data not generated
                data = np.array(session['data'])
                k = session['k']

                try:
                    k_centers = np.array(session['k_centers'])
                    print("Generated updated k_centers locations")

                except:
                    k = int(request.form.get('k', None))
                    if init_method == 'random':
                        k_centers = findRandom(data,k)
                        print("k_centers: \n", k_centers)
                        print(type(k_centers))
                    elif init_method == 'farthest':
                        k_centers = findFarthest(data,k)
                        print("k_centers: \n", k_centers)
                        print(type(k_centers))
                    elif init_method == 'kmeans++':
                        k_centers = findKMeansPlusPlus(data,k)
                        print("k_centers: \n", k_centers)
                        print(type(k_centers))
                    elif init_method == 'manual':
                        print("THIS ERROR SHOULDNT HAVE OCCURED!")

                    print("Generated New K_centers since it didn't exist before")

                
                k_centers_old = np.copy(k_centers)

                data = findClosest(data,k_centers)
                k_centers = computeCenters(data,k_centers)

               

                print("Difference:")
                diff = k_centers-k_centers_old
                print(diff)
                if np.all(diff == 0):
                    print("Converged!")
                    flash("Converged!")
                



                fig = plotDataKMeans(data,k_centers)
                print("Completed plotDataKMeans")


                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                session['data'] = data.tolist()
                session['k_centers'] = k_centers.tolist()

            if action == 'run':
                print("Running...")
                # Do run to convergence actions here
                data = np.array(session['data'])
                k = session['k']
                try:
                    k_centers = np.array(session['k_centers'])
                    print("Generated updated k_centers locations")
                except:
                    k = int(request.form.get('k', None))
                    if init_method == 'random':
                        k_centers = findRandom(data,k)
                    elif init_method == 'farthest':
                        k_centers = findFarthest(data,k)
                    elif init_method == 'kmeans++':
                        k_centers = findKMeansPlusPlus(data,k)
                     
                    print("Generated New K_centers since it didn't exist before")

                k_centers_old = np.copy(k_centers)
                data = findClosest(data,k_centers)
                k_centers = computeCenters(data,k_centers)

                diff = k_centers-k_centers_old
                print("Diff = ",diff)
                print("\n\n\nStarting Convergence Sequence...")
                print(np.all(diff == 0))
                counter = 1
                while np.all(diff == 0) == 0:
                    k_centers_old = np.copy(k_centers)
                    data = findClosest(data,k_centers)
                    k_centers = computeCenters(data,k_centers)
                    diff = k_centers-k_centers_old
                    counter = counter+1
                    print("Iteration #:",counter)

                else:
                    print("Converged!")
                    flash("Converged!")



                fig = plotDataKMeans(data,k_centers)
                print("Completed plotDataKMeans")


                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                session['data'] = data.tolist()
                session['k_centers'] = k_centers.tolist()
            
            
            if action == 'reset':
                data = np.array(session['data'])
                try:
                    session.pop('k_centers', None)
                except:
                    print("Couldn't remove k_centers since does not exist")

                data[:,2] = np.zeros(len(data))
                print(data)
                fig = plot(data)
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                session['data'] = data.tolist()

            if init_method == 'manual' and action != 'generate':
                clicked_points = request.form.get('clicked_points', [])




        except ValueError as e:
            return f"Invalid input: {e}, 400"
        
        # return jsonify({'status': 'success', 'points_received': selected_points})


    return render_template('form.html', graphJSON=graphJSON, number=number, k_value=k_value, selected_method=selected_method, clicked_points=clicked_points)



    
# def action():
#     selected_action = request.form['action']  # Identify which button was pressed
#     selected_option = request.form.get('option', None)
#     number_input = request.form.get('number_input', None)
    
#     if selected_action == 'action1':
#         # Handle Action 1 (Button 1)
#         print("Action 1 was selected")
#         # Perform some logic for Action 1
#         return render_template('form.html', action_message="Action 1 executed.")
    
#     elif selected_action == 'action2':
#         # Handle Action 2 (Button 2)
#         print("Action 2 was selected")
#         # Perform some logic for Action 2
#         return render_template('form.html', action_message="Action 2 executed.")
    
#     elif selected_action == 'action3':
#         # Handle Action 3 (Button 3)
#         print("Action 3 was selected")
#         # Perform some logic for Action 3
#         return render_template('form.html', action_message="Action 3 executed.")
    
#     return render_template('form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)