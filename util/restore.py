import tensorflow as tf;
from tensorflow.python import pywrap_tensorflow;

def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False):
    if ignore_missing_vars:
        reader = pywrap_tensorflow.NewCheckpointReader(model_path);
    if isinstance(var_list, dict):
        var_dict = var_list;
    else:
        var_dict = {var.op.name: var for var in var_list}
    available_vars = {};
    for var in var_dict:
        if reader.has_tensor(var):
            available_vars[var] = var_dict[var];
        else:
            tf.logging.warning('Variable %s missing in checkpoint %s', var, model_path);
    var_list = available_vars;
    saver = tf.train.Saver(var_list, reshape=reshape_variables);
    def callback(session):
        saver.restore(session, model_path);
    return callback;