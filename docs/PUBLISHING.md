# Publishing to GitHub

## Recommended: command line

First create an **empty** GitHub repository named `carvemix-rc`. Do not add a
README, `.gitignore`, or license on GitHub because this project already contains
them.

Then run:

```bash
cd carvemix-rc
git init
git add .
git status
git commit -m "Initial release: CarveMix-RC"
git branch -M main
git remote add origin https://github.com/<username>/carvemix-rc.git
git push -u origin main
```

GitHub no longer accepts an account password for Git operations over HTTPS. Use
a Personal Access Token when prompted, or authenticate with GitHub CLI using
`gh auth login`.

## Updating an existing repository

If `carvemix-rc` is already cloned and connected to GitHub, copy the new files
into the repository and run:

```bash
git status
git add src/postprocess/fp_aware_ensemble.py scripts/ensemble_fp_aware.sh README.md docs/METHOD.md docs/PUBLISHING.md
git diff --cached
git commit -m "Add FP-aware orientation-safe nnU-Net ensemble"
git push origin main
```

If you work on a feature branch, replace `main` with that branch name and open a
pull request after pushing.

## Browser upload

For a small update, GitHub's **Add file -> Upload files** interface can also be
used. Preserve the repository paths exactly, then commit the changes.

## After publishing

Add the repository URL to the manuscript and reviewer response if appropriate.
