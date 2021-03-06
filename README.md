# langclass
Several machine learning models for classifying languages of origin of various words, using keras, tensorflow, pandas, and scikit. Models include a 1D convolution-based approached with character embedding, a dense approach with char-embedding, a dense approach without char embedding, and a merged convolutional approach, where multiple kernel sizes and dilations are applied.

The data is from the [IDS](https://ids.clld.org/). The `clean_data.py` file will clean the appropriate data if it is in the same directory. Data cleaning written by me.

All the models except the merged model are topologically simple. Here is a graph of the merged model for convenience, generated by keras:
![](merged.png)

Below is the data for each model. These models have not been optimized, and this is likely not a fair comparison. However, it is hopeful to see the convolutional models outperforming the non-convolutional ones, and that the model without embedding performs the worst by far. I will perform some hyperparameter optimization in the future.

Merged model:
![](merged_history.png)

Convolutional model:
![](conv_history.png)

Dense model:
![](dense_history.png)

No embedding model:
![](no_embedding_history.png)
