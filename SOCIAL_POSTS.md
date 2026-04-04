# Social Sharing: TimesFM vs. Linear Projection

Here is the updated, highly engaging copy for LinkedIn (designed to be opinionated and start a conversation), plus the Threads post, both optimized for the new 4-slide carousel format.

---

## 1. LinkedIn (Opinionated & Engaging)

**Goal:** Hook the reader with a strong perspective on why simple data models fail in sports, show the AI alternative, and invite them to follow the experiment.

**The Post:**

Football isn't linear. So why do we still project end-of-season points like it is?

If you look at Chelsea right now (48 points through 31 matches), the simple math says they finish with 59 points. That’s enough to scrape into Europe. 

But anyone watching the games knows that number is lying. The form has flattened. The remaining schedule is brutal (City, Utd, Liverpool). 

I wanted to see if an AI foundation model could read that context better than a spreadsheet formula. I fed the actual match-by-match sequence into Google's TimesFM 2.5 — a 200M parameter zero-shot model built specifically for time-series forecasting. 

The model's honest read? **56 points.** 

That 3-point gap is the difference between European qualification and missing out entirely. 

The interesting part wasn't just running the model, it was how I had to frame the data to get a real answer. If you forecast a cumulative line directly, the model just predicts the "level" it ended on. You have to forecast the *per-match* points (0, 1, or 3) and stack them on the baseline. (Code in the carousel 👉)

I'm tracking this tension — Linear Math vs. Foundation Model — week over week through the end of the season. 

Which one do you trust more: the season-long average, or the AI reading the recent decay?

*(Attach the `chelsea_timesfm_carousel.pdf` file here using the "Add a document" button)*

---

## 2. Threads (Visual & Hook-Driven)

**Goal:** Quick scroll-stopper, highly visual, focused on the "what" rather than the "how."

**The Post:**

Linear math vs. AI time-series forecasting. 

Chelsea has 48 pts through 31 matches. The simple linear projection says they finish with 59 pts. 

But I ran the actual match-by-match sequence through Google's TimesFM 2.5 model. It reads the recent flat form and the brutal remaining fixtures (City, Utd, Liverpool). 

The model's honest read: 56 pts. 

That 3-point gap is the difference between Europe and nothing. Tracking it week over week through May. 

*(Attach all 4 carousel images directly to the post so people can swipe through them)*
