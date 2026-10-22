# Bharat Law Evaluation Sets

Evaluation cases are versioned with the corpus snapshot they expect. Do not
edit an existing case after a result has been published; add a new case or
version instead.

Each case records:

- the user question;
- expected source document IDs and anchors;
- an optional date/jurisdiction constraint;
- acceptable facts or an abstention requirement;
- the case category.

Run the dependency-free retrieval metric smoke check with:

```powershell
python -m evals.run_eval --dataset evals\datasets\criminal-law-smoke.json
```

The initial smoke set is intentionally small and illustrative. It is not a
legal authority and must be reviewed by a qualified legal researcher before
being used as a release gate.
