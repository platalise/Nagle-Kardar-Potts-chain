# Publishing the repo to GitHub

## 1. Personalize before pushing

Replace placeholders in these files:

| File | What to replace |
|---|---|
| `pyproject.toml` | `authors`, `Homepage`, `Paper`, `Issues` (change `OWNER` to your GitHub handle) |
| `LICENSE` | `Copyright (c) 2026 Author Name` → your name |
| `CITATION.cff` | Author name, affiliation, email, `repository-code`, `url`, arXiv id |
| `README.md` | Every `OWNER` → your GitHub handle; `XXXX.XXXXX` → your arXiv id once posted |
| `paper/manuscript.tex` | `\author`, `\affil`, `\texttt{...}` email |
| `paper/cover_letter.tex` | `\address`, `\signature` |

A single find-replace on `OWNER` and `Author Name` catches most of them:

```bash
git grep -l "OWNER\|Author Name\|author@university.edu.vn" | \
    xargs sed -i 's/OWNER/your-github-handle/g; s/Author Name/Your Real Name/g'
```

## 2. Initialize git and push

```bash
cd nkpotts                             # or wherever you unzipped the repo
git init -b main
git add .
git commit -m "Initial commit: nkpotts v0.1.0"

# Create the empty repo on github.com first (public, no README/LICENSE
# — we already have them), then:
git remote add origin git@github.com:YOUR-HANDLE/nkpotts.git
git push -u origin main
```

## 3. Enable the CI + citation button

- GitHub Actions run automatically once you push; check the **Actions** tab.
  If tests pass on 3.10 / 3.11 / 3.12, the green badge in the README goes live.
- GitHub reads `CITATION.cff` and adds a **"Cite this repository"** button to
  the sidebar automatically.

## 4. Get a citable DOI (recommended for the paper's data-availability section)

1. Log in to <https://zenodo.org> with your GitHub account.
2. In Zenodo → GitHub, flip the switch on `YOUR-HANDLE/nkpotts` to **ON**.
3. On GitHub: **Releases → Draft a new release → tag `v0.1.0`** → Publish.
4. Zenodo mints a DOI within a minute; copy the "Cite all versions" DOI and:
   - add it to the README (badge + citation block)
   - add it to `paper/manuscript.tex` in the "Data and code availability" section

The Zenodo DOI is what you cite in the paper; the GitHub URL is what people
visit to run the code.

## 5. Post to arXiv (SciPost prefers this before you submit)

Bundle:

```bash
cd paper
tar czf nkpotts-arxiv.tar.gz manuscript.tex fig_*.pdf
```

Upload to arXiv (cond-mat.stat-mech primary; cross-list cs.LG). In the comments
field write "Code and data at https://github.com/YOUR-HANDLE/nkpotts; submission
to SciPost Physics Core."

Wait ~24 h for the arXiv id, update `README.md` and `CITATION.cff` with it,
commit, push, and only then submit to SciPost — most fields on the SciPost form
auto-fill from the arXiv API.

## 6. After you have the arXiv id and Zenodo DOI

```bash
# quick swap of placeholders
sed -i "s|XXXX.XXXXX|2607.NNNNN|g" README.md CITATION.cff paper/manuscript.tex paper/cover_letter.tex
git add -A && git commit -m "Add arXiv id and Zenodo DOI" && git push
```

## Sanity checks before pushing

```bash
pip install -e ".[dev]"
pytest -q                 # 7 tests should pass in under 2 seconds
python -c "import nkpotts; print(nkpotts.__version__)"   # 0.1.0
python scripts/replot_all.py                             # rebuilds figures from data/
```
