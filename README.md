# rm-noise

### A research project to remove noise without using anything clever.

_Machine learning models rely on good datasets in order to train a good model._ This dataset, however, is often not available, hence research scientists often create innovative ways to _clean the data_. Cleaning data, however, is not always feasible, which is the reason that at the time of this research project, _data is often carefully selected_, especially those in the speech field where only very clear audio samples are used in standard datasets.

This got me thinking. What if there is a way to automatically perform cleaning on the models? The idea is not new. In fact, there were a lot of papers focussed on it. However, all of them still require clean data (for the labels) in order to train, and their model is of times not robust since they are able to handle real world noises.

In this research project, I focused on a different possibility, what if a **machine learning model by itself is able to clean the data in an unsupervised manner**? If that is the case, models no longer require perfect datasets and suddenly is capable of learning from data that do not need to be carefully chosen.

Turns out, I'm correct, as the code shows. The intuition is as follows.

It's a well known fact that machines don't learn perfectly, and the output is slightly different from the input. What I did was very simple. Train an encoder-decoder with input and output the same dirty data samples. Since the model has the same input and output shape, I can then _stack the models (with the same hyper-parameters but different parameters) in a sequential manner_, and just pass data through. _Every single model (encoder-decoder) is trained on its input._

The experiment shows that my intuition is correct if you listen to the audio samples. However, it's not obvious in the PESQ and STOI scores.
