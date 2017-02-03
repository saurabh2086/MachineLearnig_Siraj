import numpy as np

def compute_error_for_line_given_points(b, m, points):
    #initialize error to zero
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    #starting point for our gradients
    b_gradient = 0
    m_gradient = 0

    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        N = float(len(points))
        #direction with respect to b and m
        #computing partial derivatives of error function wrt to b and m
        b_gradient += -(2/N) * (y - (m_current * x + b_current))
        m_gradient += -(2/N) * x * (y - (m_current * x + b_current))
    new_b = b_current - learningRate *  b_gradient
    new_m = m_current - learningRate * m_gradient
    return [new_b, new_m]





def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b,m]


def run():
    # Step-1 Getting our data
    points = np.genfromtxt('data.csv', delimiter=",")
    
    #Step-2 Define hyperparameters
    learning_rate = 0.0001
    #y = mx + b formula for the 
    initial_b = 0
    initial_m = 0
    
    num_iteration = 1000
    
    #Step3 - Train our model
    print 'starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    [b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iteration)
    print 'ending point at b = {1}, m = {2}, error = {3}'.format(num_iteration, b, m, compute_error_for_line_given_points(b,m, points))

if __name__ == '__main__':
    run()
    