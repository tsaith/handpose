import numpy as np
import tensorflow as tf

def batch_generator(inputs, batch_size=None, shuffle=False):
    """
    Batch generator.
    """

    num_samples = len(inputs)

    if shuffle:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples - batch_size + 1, batch_size):

        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]


def eval_ops(sess, ops, feed_dict=None, batch_size=None):
    """
    Evaluate the operations.
    """

    is_list = isinstance(ops, list)

    if not is_list:
        ops = [ops]

    num_ops = len(ops)
    items = feed_dict.items()
    num_items = len(items)

    keys = []
    values = []
    for key, value in items:
        keys.append(key)
        values.append(value)

    num_samples = len(values[0])

    # Evalate the ops with batch inputs
    raw_outputs = []
    for ia in range(0, num_samples, batch_size):
        num_residual = num_samples - ia
        if num_residual < batch_size:
            iz = ia+num_residual
        else:
            iz = ia+batch_size
        batch_outputs = []
        batch_values = []
        for i in range(num_items):
            batch_values.append(values[i][ia:iz])

        batch_dict = { keys[i]:batch_values[i] for i in range(num_items) }
        batch_outputs.append(sess.run(ops, feed_dict=batch_dict))
        raw_outputs.append(batch_outputs)

    # Prepare outputs
    ops_outputs = [[] for i in range(num_ops)]
    num_batches = len(raw_outputs)
    for i in range(num_batches):
        for j in range(num_ops):
            num_elements = len(raw_outputs[i][0][j])
            for k in range(num_elements):
                ops_outputs[j].append(raw_outputs[i][0][j][k])

    outputs = [np.concatenate([ops_outputs[i]]) for i in range(num_ops)]

    if num_ops == 1:
        outputs = outputs[0]

    return outputs

