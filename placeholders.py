import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

sess = tf.Session()

print(sess.run(adder_node, {a:[1,2], b:[3,4]}))

sess.close()
