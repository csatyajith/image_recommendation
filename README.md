# image_recommendation
Image based recommendation: Massive Data Mining project

For this project, we built a Collaborative Deep Learning model to give recommendations based on Images. 

The paper we used as a primary reference is https://arxiv.org/pdf/1409.2944.pdf

First, we used the Amazon reviews dataset to compute our recommendations. As the dataset is several GB in size, 
we used just the Baby sub-category from 2014. This gave us a miniature platform to experiment our model.
The dataset link can be found here: https://snap.stanford.edu/data/amazon/productGraph/

We took inspiration from the following repository to build our model using Keras. We made several changes as our dataset is entirely different. 
But the repository here provided a valuable start when it came to getting a structure and the default hyperparameters: 
https://github.com/js05212/CollaborativeDeepLearning-TensorFlow

Steps to run our code:
Step 1: Download everything in this google drive link and paste it in the analysis2014 folder: 
https://drive.google.com/drive/folders/1qeGUrHtF_-2m5yW1I0Qs2qOiEFrMPuXY?usp=sharing

Step 2: Run the file additional_work.ipynb. 

Step 3: The estimated run time is approximately 25 minutes. Alternatively, you can take a look at the results already obtained in the following 2 files:
i) main.ipynb
ii) additional_work.ipynb

