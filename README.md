<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the last week of the cohort starts.


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

### 🔐 Accessing private GitHub dependencies

This project depends on a private repository called `marvelous`.  
The source is configured in `pyproject.toml` like this:

```toml
[tool.uv.sources]
marvelous = { git = "https://x-access-token:${GIT_TOKEN}@github.com/end-to-end-mlops-databricks-3/marvelous.git@main" }
```

#### Steps to configure:

1. Go to [GitHub → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens)
2. Generate a new **classic** token (not fine-grained)
3. Copy the token
4. Add it to your environment:

   ```bash
   export GIT_TOKEN=your_token_here
   ```

To create a new environment and create a lockfile, run:

```
uv venv -p 3.11 .venv
source .venv/bin/activate
uv sync --extra dev
```



# Data
Using the [**House Price Dataset**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) from Kaggle.

This data can be used to build a classification model to calculate house price.

