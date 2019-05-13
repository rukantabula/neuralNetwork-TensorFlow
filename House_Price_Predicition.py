
import tensorflow as tf 
import numpy as np # random number generation and array convertion features
import math # math functions
import matplotlib.pyplot as plt # animate and plot the charts
import matplotlib.animation as animation


# generate some random house sizes between 1000 and 1500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=100, high=3500, size=num_house)

# generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * np.random.randint(low=20000, high=70000, size=num_house)

# plot generates house and size

plt.plot(house_size, house_price, "bx") # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

# normalize values to prevent under/overflows

def normalize(array):
    return (array - array.mean()) / array.std()

# PREPARE DATA

# define number of training samples, 0.7 = 70%.  
num_train_samples = math.floor(num_house * 0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)


# Set up TensorFlow placeholders that get updated as we descnd down the gradiens
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# Define the variables holding the size_factor and price we set during training
# initialize them to some random values based on the normal distribution

tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

#2. Define the operations for the predicting values

tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

#3. Define the loss function (how much error) - Mean square error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

# Optimizer learing rate. The size of steps down the gradienr
learning_rate = 0.1

#4. deifne gradient descent optimizer that will minimize the loss defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initializing the variables

init = tf.global_variables_initializer()

#Launch the graph in the session
with tf.Session() as sess:
    sess.run(init)
    

    # set how often to display training progress and number of training iterations
    display_every = 2
    num_training_iter = 50

    # keep iterating the data
    for iteration in range(num_training_iter):

        # fit all training data
        for (x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

         # Display current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price:train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
            "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Opimization Finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "Size_factor=", sess.run(tf_size_factor), "price_offset", sess.run(tf_price_offset))

    train_house_size_std = train_house_size.std()
    train_house_size_mean = train_house_size.mean()
    train_price_std = train_price.std()
    train_price_mean = train_price.mean()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
            (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
            label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()        
       