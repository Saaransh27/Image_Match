## Whats the purpose of this project?

This interactive program takes an image from the user as an input and outputs the most similar image it can find from the input folder given to it

# There are two types...

We have two similar files Light.py and Heavy.py but both have different workings.
Light.py asks for an input folder and would processes each image in the input folder each time the code is run.

# Use Light.py for small folder

However the there is another file preprocess_images.py which can be used to preprocess a bulky folder if has to be used many times.

# Use Heavy.py for big folders

Heavy.py accepts the file containing the preprocessed images created by preprocess_images.py with the name "image_encodings.pkl" and gives the output in mere seconds.