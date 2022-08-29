import numpy as np
from os import path

import loader

if __name__ == "__main__":
    tp_posteriors = np.load(path.abspath("tp_posteriors.npy"))
    tp_labels = np.load(path.abspath("tp_labels.npy"))
        
    tp_posteriors = np.expand_dims(tp_posteriors, axis=1) #para incorporar el canal (que es uno solo en este caso)

    ta_posteriors = np.load(path.abspath("ta_posteriors.npy"))
    ta_labels = np.load(path.abspath("ta_labels.npy"))
        
    ta_posteriors = np.expand_dims(ta_posteriors, axis=1) #para incorporar el canal (que es uno solo en este caso)

    
    posteriors = np.concatenate((tp_posteriors,ta_posteriors),axis=0)
    labels = np.concatenate((tp_labels,ta_labels),axis=0)
    del ta_posteriors,ta_labels,tp_posteriors,tp_labels

    model_loader = loader.ModelLoader()
    model_loader.fit(posteriors,labels)

