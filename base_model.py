class BaseModel():
    def __init__(self, logger, gpu_id, learning_rate, input_dim, z_dim):
        self.logger = logger
        self.gpu_id = gpu_id

        self.learning_rate = learning_rate 

        self.input_dim = input_dim
        self.z_dim = z_dim

