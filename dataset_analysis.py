import numpy as np
from Models import Models

models = Models()
np_comments, np_emotions, np_sentiments = models.get_dataset()

n = len(np_comments)
print(f"{n:,} Comments")

def calculate_freq(data, class_type):
    print(f"\n{class_type}: ") 
    return np.unique(data, return_counts=True)

def display_frequency_breakdown(lab, freq):
    for l,c in zip(lab, freq):
        print(f" - {l}: {c:,} ({round(c/n * 100, 1)})")

def calculate_analytics(freq): 
    print("\nMean: ", np.mean(freq))
    print("Variance: ", np.var(freq))
    print("Standard Deviation: ", np.std(freq))


# Calculate analytics for emotions
emo_lab, emo_freq = calculate_freq(np_emotions, "Emotions")
display_frequency_breakdown(emo_lab, emo_freq)
calculate_analytics(emo_freq)

# Calculate analytics for sentiments
sent_lab, sent_freq = calculate_freq(np_sentiments, "Sentiments")
display_frequency_breakdown(sent_lab, sent_freq)
calculate_analytics(sent_freq)


