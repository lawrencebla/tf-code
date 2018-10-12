import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

result = tf.add(a, b, name="add")
print result
print a + b

sess = tf.Session()
print(sess.run(result))
sess.close()

with tf.Session() as sess:
    print(sess.run(result))
