# Social Sharing: TimesFM vs. Linear Projection

Here is the updated, conversational copy for LinkedIn (framing it as "TimesFM exists, applied it, here are some thoughts"), plus the Threads post. The `TIMESFM_LIVE_SNAPSHOT.md` document remains intact in the repo.

---

## 1. LinkedIn (Conversational & Applied)

**Goal:** A lighter, "build in public" tone. Introduce TimesFM, show a real-world application on sports data, and share a few technical takeaways from the experiment.

**The Post:**

Google recently released TimesFM 2.5 — a 200M parameter zero-shot foundation model built specifically for time-series forecasting. 

I wanted to see how it handles short, noisy sequences, so I applied it to Chelsea's current Premier League season. 

Right now, Chelsea has 48 points through 31 matches. If you run a simple linear projection (current pace * 38 games), they finish with 59 points. But when you feed the actual match-by-match sequence into TimesFM, it reads the recent flat form and the brutal remaining schedule differently. 

The model's read: 56 points. 

A few thoughts from building this out:

1. **Foundation models for time-series are getting very accessible.** You don't need to fine-tune this on football data. You just feed it an array of numbers and it forecasts forward with an 80% confidence interval.
2. **Framing the data is everything.** If you forecast a cumulative points line directly, the model predicts a flat "level" if the recent games were losses. You have to forecast the *per-match* points (0, 1, or 3) and stack them on the baseline to get a real growth projection.
3. **The tension is trackable.** We now have a 3-point gap between simple math (59) and the AI read (56). That's the difference between European qualification and missing out.

I put together a quick carousel on how the implementation works, and I'll be tracking the "Linear vs. TimesFM" gap week over week through May. 

Full code and live tracker in the repo: [Link to your GitHub repo]

*(Attach the `chelsea_timesfm_carousel.pdf` file here using the "Add a document" button)*

---

## 2. Threads (Visual & Hook-Driven)

**Goal:** Quick scroll-stopper, highly visual, focused on the "what" rather than the "how."

**The Post:**

Applied Google's TimesFM 2.5 model to Chelsea's season. 

Simple math says they finish with 59 pts. 
The AI model reads the recent flat form and brutal remaining schedule. Its read: 56 pts. 

That 3-point gap is the difference between Europe and nothing. Tracking it week over week through May. 

*(Attach all 4 carousel images directly to the post so people can swipe through them)*
