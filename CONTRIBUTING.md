# Contributing Guide

Thanks for thinking about contributing!

Please only make one change or add one feature per Pull Request (PR). Limit the scope of your PR so that it makes small, manageable size changes. From opening your branch to merging it should take maximum 2 weeks. Large PRs may not be accepted.

To help us integrate your changes, please follow our standard process: 

## Development Process:

1. Make a new issue (or use an existing one). You will need the issue number when you create a branch.
2. Clone this repo
    - `git clone https://github.com/AstraZeneca/chemicalx`
3. Create a new branch. The branch name must include the issue number.
    - `git checkout main`
    - `git branch <your branch name>`
    - `git checkout <your branch name>`
4. Install **ChemicalX** in development mode. This also sets up pre-commits for you.
    - `./dev_setup.sh`
5. Make your changes in small, logically grouped commits (this makes it easier to review changes):
    - Document your code as you go.
    - Add unit and integration tests for your code.
6. Run tests.
    - Run unit tests: `pytest --cov chemicalx/tests/unit`
    - Run integration tests: `pytest --cov chemicalx/tests/integration`
    - Make sure that your contributions have test coverage.
7. Update the documentation with your changes.
    - Documentation is located in docs/.
    - Add new classes or modules to API documents.
    - Add new/changed functionality to the tutorials or quickstart.
    - Add code snippets (these are tested when docs are built so make them small and quick to run).
8. When finished, make a Pull Request (PR).
    - Use our PR template located at: ./pull_request_template.md.
    - Describe change and clearly highlight any major or breaking changes.
    - If any errors occur on the test builds, please fix them.
    - You are responsible for getting your PR merged so please chase down your reviewers.
9. Adjust your PR based on any feedback
    - We use the [DO, TRY, CONSIDER](https://jackiebo.medium.com/do-try-consider-how-we-give-product-feedback-at-asana-db9bc754cc4a) framework to give constructive feedback.
10. After approval, you are responsible for completing your PR .