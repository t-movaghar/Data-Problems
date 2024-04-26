def get_ds_size(folder_path): #To get the size of an image dataset
    
    import os
    
    num_of_images = {} 
    for folder in os.listdir(folder_path): #Needed a method to count the distribution of my classes.
        num_of_images[folder] = len(os.listdir(os.path.join(folder_path, folder)))
    return num_of_images

def show_images(gen): #prints out a sample of images within a dataset, labeled.
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())     # defines list of dictionary's kays (classes)
    images, labels = next(gen)        # get a batch size samples from the generator
    plt.figure(figsize= (20, 20))
    length = len(labels)              # length of batch size
    sample = min(length, 25)          # check if sample less than 25 images
    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] #modified this. used to rescale data, but mine is already scaled...
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]   # get class of image
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()

def data_loader(datagen, directory_path): #a data loader function that takes in a datagen and a particular directory
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    dataset = datagen.flow_from_directory(directory = directory_path,
                                                   target_size = (100,100),
                                                   class_mode = 'categorical',
                                                   batch_size = 128)
    return dataset