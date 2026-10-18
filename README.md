# ASX/NZX Annual Reports Data Extraction App

A Streamlit app that uses Claude to extract and classify information from Australian (ASX) and New Zealand (NZX) company annual reports at scale.

You describe what you want from each report, such as the company's industry or its revenue, and the app runs that prompt over the report sections you choose. The database covers about 2,150 companies for the financial years 2023–2025. Each answer comes with the quote from the report that supports it. You can try your prompt on 15 random reports and check the estimated cost before running it on everything, then download the results as a CSV.

Each answer can also be checked by TypeSafe's Jev verifier, which gives it a confidence score.

## What you need

- A Claude API key from Anthropic. Usage is billed to your own account.
- A TypeSafe API key, only if you want confidence scores.

You enter your keys in the app. They are used only during your session and are never saved.
