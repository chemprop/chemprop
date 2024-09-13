import pickle


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "MSELoss":
            name = "MSE"
        elif name == "BoundedMSELoss":
            name = "BoundedMSE"
        elif name == "SIDLoss":
            name = "SID"
        elif name == "WassersteinLoss":
            name = "Wasserstein"
        return super().find_class(module, name)
