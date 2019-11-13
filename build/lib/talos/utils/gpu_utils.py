def parallel_gpu_jobs(allow_growth=True, fraction=.5):

    '''Sets the max used memory as a fraction for tensorflow
    backend

    allow_growth :: True of False

    fraction :: a float value (e.g. 0.5 means 4gb out of 8gb)

    '''

    import keras.backend as K
    import tensorflow as tf
    from nvidia_info import get_memory_info

    memory_info = get_memory_info(0)
    total_memory = memory_info[1]
    memory_limit = int(fraction*total_memory)
    print(memory_info)
    if tf.version.VERSION[0]=="2":
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    else:
        gpu_options = tf.GPUOptions(allow_growth=allow_growth,
                                  per_process_gpu_memory_fraction=fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
        session = tf.Session(config=config)
        K.set_session(session)


def multi_gpu(model, gpus=None, cpu_merge=True, cpu_relocation=False):

    '''Takes as input the model, and returns a model
    based on the number of GPUs available on the machine
    or alternatively the 'gpus' user input.

    NOTE: this needs to be used before model.compile() in the
    model inputted to Scan in the form:

    from talos.utils.gpu_utils import multi_gpu
    model = multi_gpu(model)

    '''

    from keras.utils import multi_gpu_model

    return multi_gpu_model(model,
                           gpus=gpus,
                           cpu_merge=cpu_merge,
                           cpu_relocation=cpu_relocation)


def force_cpu():

    '''Force CPU on a GPU system
    '''

    import keras.backend as K
    import tensorflow as tf

    config = tf.ConfigProto(device_count={'GPU': 0})
    session = tf.Session(config=config)
    K.set_session(session)
