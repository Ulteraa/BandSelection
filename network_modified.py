def vgg_16(train_x, variables_dict):
    inputs = tf.cast(train_x, tf.float32)

    conv = tf.nn.conv2d(inputs, variables_dict['conv1_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv1_1_biases'])

    conv1_1 = tf.nn.relu(bias, name='conv1_1')

    conv = tf.nn.conv2d(conv1_1, variables_dict['conv1_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv1_2_biases'])

    conv1_2 = tf.nn.relu(bias, name='conv1_2')

    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    conv = tf.nn.conv2d(pool1, variables_dict['conv2_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv2_1_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv2_1 = tf.nn.relu(bias, name='conv2_1')

    conv = tf.nn.conv2d(conv2_1, variables_dict['conv2_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv2_2_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv2_2 = tf.nn.relu(bias, name='conv2_2')

    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    conv = tf.nn.conv2d(pool2, variables_dict['conv3_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv3_1_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv3_1 = tf.nn.relu(bias, name='conv3_1')

    conv = tf.nn.conv2d(conv3_1, variables_dict['conv3_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv3_2_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv3_2 = tf.nn.relu(bias, name='conv3_2')

    conv = tf.nn.conv2d(conv3_2, variables_dict['conv3_3_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv3_3_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv3_3 = tf.nn.relu(bias, name='conv3_3')

    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    conv = tf.nn.conv2d(pool3, variables_dict['conv4_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv4_1_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv4_1 = tf.nn.relu(bias, name='conv4_1')

    conv = tf.nn.conv2d(conv4_1, variables_dict['conv4_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv4_2_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv4_2 = tf.nn.relu(bias, name='conv4_2')

    conv = tf.nn.conv2d(conv4_2, variables_dict['conv4_3_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv4_3_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv4_3 = tf.nn.relu(bias, name='conv4_3')

    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    conv = tf.nn.conv2d(pool4, variables_dict['conv5_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv5_1_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv5_1 = tf.nn.relu(bias, name='conv5_1')

    conv = tf.nn.conv2d(conv5_1, variables_dict['conv5_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv5_2_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv5_2 = tf.nn.relu(bias, name='conv5_2')

    conv = tf.nn.conv2d(conv5_2, variables_dict['conv5_3_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv5_3_biases'])

    conv5_3 = tf.nn.relu(bias, name='conv5_3')

    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    conv = tf.nn.conv2d(pool5, variables_dict['fc6_weights'], [1, 1, 1, 1], padding='VALID')

    bias = tf.nn.bias_add(conv, variables_dict['fc6_biases'])

    fc6 = tf.nn.relu(bias, name='fc6')
    conv = tf.nn.conv2d(fc6, variables_dict['fc7_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['fc7_biases'])

    fc7 = tf.nn.relu(bias, name='fc7')
    conv = tf.nn.conv2d(fc7, variables_dict['fc8_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['fc8_biases'])
    fc8 = tf.nn.relu(bias, name='fc8')
    fc8 = tf.squeeze(bias, [1, 2], name='fc8')

    '''
    shape = pool5.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(pool5, [-1, dim])
 
    fc6 = tf.nn.bias_add(tf.matmul(x, variables_dict['fc6_weights']), variables_dict['fc6_biases'])
    fc6 = tf.nn.relu(fc6, name='fc6')
    fc7 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc7_weights']), variables_dict['fc7_biases'])
    fc7 = tf.nn.relu(fc7, name='fc7')
    # dropout = tf.layers.dropout(inputs= fc7, rate=0.5, training=phase)
    # fc8 = tf.nn.bias_add(tf.matmul(dropout, variables_dict['fc8_weights']), variables_dict['fc8_biases'])
    fc8 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc8_weights']), variables_dict['fc8_biases'])
    # fc8  = tf.contrib.layers.batch_norm(fc8 , center=True, scale=True, is_training=phase)
    '''

    return fc8


def vgg_19(image, variables_dict):
    # for d in ['/gpu:1', '/gpu:0']:
    # with tf.device(d):
    inputs = tf.cast(image, tf.float32)

    conv = tf.nn.conv2d(inputs, variables_dict['conv1_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv1_1_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)

    conv1_1 = tf.nn.relu(bias, name='conv1_1')

    conv = tf.nn.conv2d(conv1_1, variables_dict['conv1_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv1_2_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)

    conv1_2 = tf.nn.relu(bias, name='conv1_2')

    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    conv = tf.nn.conv2d(pool1, variables_dict['conv2_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv2_1_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv2_1 = tf.nn.relu(bias, name='conv2_1')

    conv = tf.nn.conv2d(conv2_1, variables_dict['conv2_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv2_2_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv2_2 = tf.nn.relu(bias, name='conv2_2')

    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    conv = tf.nn.conv2d(pool2, variables_dict['conv3_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv3_1_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv3_1 = tf.nn.relu(bias, name='conv3_1')

    conv = tf.nn.conv2d(conv3_1, variables_dict['conv3_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv3_2_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv3_2 = tf.nn.relu(bias, name='conv3_2')

    conv = tf.nn.conv2d(conv3_2, variables_dict['conv3_3_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv3_3_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv3_3 = tf.nn.relu(bias, name='conv3_3')

    conv = tf.nn.conv2d(conv3_3, variables_dict['conv3_4_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv3_4_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv3_4 = tf.nn.relu(bias, name='conv3_4')

    pool3 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    conv = tf.nn.conv2d(pool3, variables_dict['conv4_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv4_1_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv4_1 = tf.nn.relu(bias, name='conv4_1')

    conv = tf.nn.conv2d(conv4_1, variables_dict['conv4_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv4_2_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv4_2 = tf.nn.relu(bias, name='conv4_2')

    conv = tf.nn.conv2d(conv4_2, variables_dict['conv4_3_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv4_3_biases'])

    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv4_3 = tf.nn.relu(bias, name='conv4_3')

    conv = tf.nn.conv2d(conv4_3, variables_dict['conv4_4_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv4_4_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv4_4 = tf.nn.relu(bias, name='conv4_4')
    pool4 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    conv = tf.nn.conv2d(pool4, variables_dict['conv5_1_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv5_1_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv5_1 = tf.nn.relu(bias, name='conv5_1')

    conv = tf.nn.conv2d(conv5_1, variables_dict['conv5_2_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv5_2_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv5_2 = tf.nn.relu(bias, name='conv5_2')

    conv = tf.nn.conv2d(conv5_2, variables_dict['conv5_3_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv5_3_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv5_3 = tf.nn.relu(bias, name='conv5_3')
    conv = tf.nn.conv2d(conv5_3, variables_dict['conv5_4_weights'], [1, 1, 1, 1], padding='SAME')

    bias = tf.nn.bias_add(conv, variables_dict['conv5_4_biases'])
    # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
    conv5_4 = tf.nn.relu(bias, name='conv5_4')
    pool5 = tf.nn.max_pool(conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    shape = pool5.get_shape().as_list()

    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(pool5, [-1, dim])
    fc6 = tf.nn.bias_add(tf.matmul(x, variables_dict['fc6_weights']), variables_dict['fc6_biases'])

    fc6 = tf.nn.relu(fc6, name='fc6')

    fc7 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc7_weights']), variables_dict['fc7_biases'])

    # fc7 = tf.nn.relu(fc7, name='fc7')

    # fc8 = tf.nn.bias_add(tf.matmul(fc7, variables_dict['fc8_weights']), variables_dict['fc8_biases'])

    # fc8 = tf.nn.relu(fc8, name='fc8')
    '''
    conv = tf.nn.conv2d(pool5, variables_dict['fc6_weights'], [1, 1, 1, 1], padding='VALID')
    bias = tf.nn.bias_add(conv, variables_dict['fc6_biases'])
 
    fc6 = tf.nn.relu(bias, name='fc6')
 
    conv = tf.nn.conv2d(fc6, variables_dict['fc7_weights'], [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, variables_dict['fc7_biases'])
    fc7 = tf.nn.relu(bias, name='fc7')
 
    conv = tf.nn.conv2d(fc7 , variables_dict['fc8_weights'], [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, variables_dict['fc8_biases'])
    fc8 = tf.nn.relu(bias, name='fc8')
    fc8 = tf.squeeze(fc8, [1, 2], name='fc8')
 
    print(fc8.shape)
    return fc8
    '''