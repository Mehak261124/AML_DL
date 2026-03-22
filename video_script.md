# AML/DL Phase 1 - 10 Minute Presentation Script
**Target Length:** 8 to 10 minutes
**Project:** Dynamic Trend & Event Detector
**Author:** Mehak Jain

---

## Slide 1 / Intro (0:00 - 1:00) | Motivation
*(Show Title Slide with your Topic)*
"Hello everyone, I'm Mehak Jain, and my project is the **Dynamic Trend & Event Detector: Emerging Topic Detection & News Correlation**."
**Motivation:** "The problem I wanted to solve was distinguishing between a fleeting, superficial social media meme and a massive, structural real-world news event as it breaks in real time. Standard probabilistic topic models like Latent Dirichlet Allocation (LDA) are fantastic at finding themes, but they suffer from a massive mathematical flaw: **They are completely static**. They treat an article from 2012 identically to an article from 2022. They completely ignore temporal momentum. My goal was to fix this by engineering what I call 'Semantic Velocity'."

---

## Slide 2 / Demo Setup & EDA (1:00 - 3:00) | The Data
*(Switch screen to your Code Editor / Demo)*
"Let's jump straight into the code. The pipeline is fully reproducible."
*(Run `python3 main.py` in your terminal)*
"While this runs, I'm processing an 11,000-article subset of the HuffPost News Category Dataset spanning from 2012 to 2022. I specifically engineered the script to sample exactly 1,000 articles per year. Why? **To prevent global volume inflation.** If I didn't balance it, election years would artificially skew all the velocity curves."

*(Open `eda_plots/category_velocity.png`)*
"My Exploratory Data Analysis immediately revealed a non-obvious pattern that dictated my whole strategy: You can see right here that the 'Politics' category grew by exactly 278% between 2016 and 2017. This proved that real-world events deeply warp dataset structures. We needed a model to capture that momentum."

---

## Slide 3 / The Baseline (3:00 - 4:00) | TF-IDF
*(Open `model_plots/baseline_tfidf.png`)*
"Initially, I ran a standard TF-IDF baseline extraction. The word 'Trump' scored the absolute highest across the entire 11-year corpus. But this highlighted the limit of static models: TF-IDF treats the entire 11 years as one massive block. It doesn't tell us *when* a trend emerged or *how fast* it grew. This led me to my core breakthrough."

---

## Slide 4 / The Breakthrough (4:00 - 6:00) | Feature Engineering
*(Open `model_plots/feature_engineering.png`)*
"I engineered three novel features to inject time into LDA calculations. This is the core algorithmic contribution."
"**First — Temporal TF-IDF Weighting:** I applied a logarithmic recency penalty. I used logarithms specifically because linear decay drops older articles too fast. This forces the model to weigh recent documents higher, actively breaking LDA's assumed sequence exchangeability."
"**Second — Category Velocity Score:** Before I fit the model, I calculate how fast the article's core category was natively surging the exact month it was published."
"**Third — Text Richness Ratio:** I scaled the unique-to-total word density to give more informative articles higher impact."
*(Show Terminal)*
"I multiply this resulting vector through a sparse diagonal weight matrix before fitting LDA. Let's run `lda_model.py`."
*(Run `python3 lda_model.py`)*

---

## Slide 5 / The Results (6:00 - 7:30) | Tracking Events
*(Let the LDA finish running on screen. Open `model_plots/lda_topics.png` AND `model_plots/lda_topic_evolution.png`)*
"As you can see, 10 topics were discovered entirely unsupervised. Our temporally weighted matrix immediately improved topic coherence and confidence scores."
"Most importantly, look at the temporal tracking. The algorithm organically clusters the 'Trump/President' topic and mathematically spikes it right during the 2014-2016 campaign cycle precisely mimicking my EDA."
"Even better, we completely unsupervised-discovered a 'COVID / Coronavirus' topic purely from the word distributions."

---

## Slide 6 / Failure Analysis (7:30 - 9:00) | Theoretical Rigor
*(Keep the LDA Topics image open highlighting the COVID topic)*
"However, I want to perform a rigorous failure analysis. In my tracking output, the COVID topic actually maxed its semantic velocity tracking in **2014** — six years before COVID existed. Why did my model hallucinate?"
"When I dug into the data mathematically, I realized LDA's 'Bag of Words' assumption collapsed the spatial context."
"In 2014, the massive Affordable Care Act legal battles flooded the news heavily using the words `state, health, court, people`. 
In 2020, COVID-19 flooded the news heavily using `state, health, court, people`."
"Because LDA throws away sentence structure, the Cosine Similarity between a 2014 Legal Health article and a 2020 Viral Health article is near exactly 1. They conflated."

---

## Slide 7 / Conclusion & Phase 2 (9:00 - 10:00) | Wrapping Up
*(Show your repo README again)*
"To conclude: We successfully built an algorithmic pipeline that tracks chronological momentum, solving LDA's static time assumption natively through Novel Feature Engineering. We proved it by extracting unsupervised political surges."
"But our failure analysis formally proved that Bag-of-Words limits eventually conflate contexts. Therefore, Phase 2 of this project is explicitly mapping our Temporal Weights away from CountVectorizers, and into high-dimensional space using **Sentence-BERT embeddings and BERTopic clustering**, allowing us to mathematically separate *'Court rulings on health'* from *'Coronavirus health protocols'*. All code is up on my GitHub. Thank you!"
