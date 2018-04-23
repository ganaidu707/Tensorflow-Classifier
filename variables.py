import tensorflow as tf

#  model parameters
w = tf.Variable([-.1], tf.float32)
b = tf.Variable([.1], tf.float32)

x = tf.placeholder(tf.float32)

# model inputs and outputs
linear_model = w * x + b

y = tf.placeholder(tf.float32)

# loss
squared_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_delta)

# optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(10000):
    sess.run(train, {x:[1, 2, 3, 4], y:[-0, -1, -2, -3]})

# print(sess.run(loss, {x:[1, 2, 3, 4], y:[-0, -1, -2, -3]}))
print(sess.run([w, b]))

sess.close()
