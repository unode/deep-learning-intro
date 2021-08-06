---
title: "Instructor Notes"
---

## Episode 3: Monitor the training process
When episode 3 is taught on a different day then episode 2, it is very useful to start with a recap of episode 2. This will help learners in the big exercise on creating a neural network.

If learners did not download the data yet, they can also load the data directly from zenodo (instead of first downloading and saving):
~~~
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")
~~~
{: .language-python}

The following exercises work well to do in groups / break-out rooms:
- Split data into training, validation, and test set
- Create the neural network. Note that this is a fairly challenging exercise, but learners should be able to do this based on their experiences in episode 2 (see also remark about recap).
- Predict the labels for both training and test set and compare to the true values
- Try to reduce the degree of overfitting by lowering the number of parameters
- Create a similar scatter plot as above for a reasonable baseline
- Open question: What could be next steps to further improve the model?
All other exercises are small and can be done individually.


{% include links.md %}
