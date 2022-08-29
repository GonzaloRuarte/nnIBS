import numpy as np
from os import path

import loader

if __name__ == "__main__":
    tp_posteriors = np.load(path.abspath("tp_posteriors.npy"))
    tp_labels = np.load(path.abspath("tp_labels.npy"))
        
    tp_posteriors = np.expand_dims(tp_posteriors, axis=1) #para incorporar el canal (que es uno solo en este caso)

    model_loader = loader.ModelLoader()
    model_loader.fit(tp_posteriors,tp_labels)

