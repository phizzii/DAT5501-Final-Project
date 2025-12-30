# my model is a review-level analytical framework to detect early signs of negative user experience and platform-specific bias, by analysing linguistic and sentiment features in user-generated reviews, enabling proactive reputation and product management
# hypothetically, a company would be able to monitor INCOMING reviews automatically, detect emerging dissatisfaction patterns early, understand how negativity is expressed differently per platform (diff user bases) and act before average ratings or sales metrics visibly decline

# le plan
# discrete outcome (negative experience) defined as 'rating' being: <= 2 (negative), >= 4 (positive) and == 3 (neutral)

# feature groups

# putting raw text isn't really plausible so there's going to be sentiment features which are interpretability focused like the proportion of negative words used (business meaning being how emotionally negative is the review?)
# linguistic intensity features, like review length (word count), maybe number of symbols used (like exclamation marks), how much capitalisation is there?, use of intensifiers like 'very', 'extremely' or 'worst' (business meaning being how strongly is the user expressing negativity?)
# structural features, like number of stars or how many people found the review useful?