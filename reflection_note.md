# Reflection Note

With more data, I'd expand the training set well beyond 40 rows (200+ per
category) — the small dataset is why a fair number of the sample tickets
land below the 60% confidence threshold; more examples per category would
sharpen the model's word-probability estimates and raise confidence on
correctly-classified tickets. I'd also add a proper "Uncategorized/Other"
class instead of only flagging low confidence, since some real tickets
genuinely don't fit Billing/Technical/HR/General. With more time, I'd
compare TF-IDF against word embeddings (e.g. sentence-transformers) for
semantic similarity rather than exact keyword overlap, and I'd build a
small labeled validation set specifically of ambiguous/edge-case tickets to
tune the review threshold more rigorously than the default 60% guess.
