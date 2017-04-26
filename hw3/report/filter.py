import tensorflow as tf
import numpy as np
np.random.seed(3318)  # for reproducibility
tf.set_random_seed(3318)

filters = []
with open("weights.txt", "r") as fin:
    for line in fin:
        filters.append( list( map( float, line.rstrip().split() ) ) )
filters = np.array(filters).reshape(64, 3, 3)

fout = open("activate.txt", "w")
for i in range(64):
    flt_reshape = tf.placeholder(tf.float32, [9])
    img = tf.Variable( tf.random_uniform([48, 48], 0, 1) )
    total_cost = 0.
    for row in range(45):
        for col in range(45):
            img_reshape = tf.reshape(img[row : row + 3, col : col + 3], [-1])
            cost = tf.maximum(tf.multiply(flt_reshape, img_reshape), 0)
            total_cost = total_cost + cost
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(-total_cost)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        sess.run(init)
        for step in range(200):
            sess.run(optimizer, feed_dict = {flt_reshape: filters[i, :, :].reshape(-1)})
        result_img = sess.run(img).reshape(-1)
        for value in result_img:
            fout.write(str(value) + " ")
        fout.write("\n")
    print(str(i + 1) + "/64")
fout.close()

