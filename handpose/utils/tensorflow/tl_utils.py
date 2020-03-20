import tensorflow as tf
import tensorlayer as tl


def model_test(sess, network, X_test, Y_test, x_, y_, cost=None, acc=None, batch_size=None):
    """
    Test the trained model.
    """
    test_loss, test_acc, batches = 0, 0, 0
    for X_batch, Y_batch in tl.iterate.minibatches(X_test, Y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(network.all_drop) # disable dropout
        feed_dict = {x_: X_batch, y_: Y_batch}
        feed_dict.update(dp_dict)
        loss_batch, acc_batch = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += loss_batch; test_acc += acc_batch; batches += 1
    test_loss /= batches
    test_acc /= batches

    return test_loss, test_acc


