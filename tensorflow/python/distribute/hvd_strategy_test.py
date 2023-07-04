from tensorflow.python.distribute import hvd_strategy

if __name__ == "__main__":
    hvd_context = hvd_strategy.HvdContext()