# How to contribute to Eurybia Open source

This guide aims to help you contributing to Eurybia. If you have found any problems, improvements that can be done, or you have a burning desire to develop new features for Eurybia, please make sure to follow the steps bellow.

- [How to open an issue](#how-to-open-an-issue)
- [Create your contribution to submit a pull request](#create-your-contribution-to-submit-a-pull-request)
    - [Fork to code in your personal Eurybia repo](#fork-to-code-in-your-personal-eurybia-repo)
    - [Clone your forked repository](#clone-your-forked-repository)
    - [Make sure that your repository is up to date](#make-sure-that-your-repository-is-up-to-date)
    - [Start your contribution code](#start-your-contribution-code)
    - [Commit your changes](#commit-your-changes)
    - [Create a pull request](#create-a-pull-request)
    - [Finally submit your pull request](#finally-submit-your-pull-request)

# How to open an issue

**Screenshots are coming soon**

An issue will open a discussion to evaluate if the problem / feature that you submit is eligible, and legitimate for Eurybia.

Check on the project tab if your issue / feature is not already created. In this tab, you will find the roadmap of Eurybia.

A Pull Request must be linked to an issue.
Before you open an issue, please check the current opened issues to insure there are no duplicate. Define if it's a feature or a bugfix.

Next, the Eurybia team, or the community, will give you a feedback on whether your issue must be implemented in Eurybia, or if it can be resolved easily without a pull request.

# Create your contribution to submit a pull request
## Fork to code in your personal Eurybia repo

The first step is to get our MAIF repository on your personal GitHub repositories. To do so, use the "Fork" button.

<img src="https://github.com/MAIF/eurybia/blob/master/docs/assets/images/contributing/eurybia-fork.png" alt="fork this repository" />

## Clone your forked repository

<img align="right" width="300" src="https://github.com/MAIF/eurybia/blob/master/docs/assets/images/contributing/eurybia-clone.png" alt="clone your forked repository" />

Click on the "Code" button to copy the url of your repository, and next, you can paste this url to clone your forked repository.

```
git clone https://github.com/YOUR_GITHUB_PROFILE/eurybia.git
```

## Make sure that your repository is up to date

To insure that your local forked repository is synced, you have to update your repo with the master branch of Eurybia (MAIF). So, go to your repository and as follow :

```
cd Eurybia
git remote add upstream https://github.com/MAIF/eurybia.git
git pull upstream master
```

## - Set up your development environment

- Install the development requirements
```
pip install -r ./requirements.dev.txt
```

- Set up the **pre-commit hooks** in your local copy of Eurybia
```
pre-commit install
```

## Start your contribution code

To contribute to Eurybia, you will need to create a personal branch.
```
git checkout -b feature/my-contribution-branch
```
We recommend using the following convention for naming branches
- **feature/your_feature_name** if you are creating a feature
- **hotfix/your_bug_fix** if you are fixing a bug

## Check and commit your changes

Before committing your modifications, we have some recommendations :

- Execute pytest to check that all tests pass
```
pytest
```
- Try to build Eurybia
```
python setup.py bdist_wheel
```
- Stage your modifications
```
git add .
```
- Commit your changes : **This will execute the pre-commit hooks, possibly modifying some files**
We recommend committing with clear messages and grouping your commits by modifications dependencies.
```
git commit -m ‘fixed a bug’
```
If the pre-commit hooks modify some of your files, add and commit those changes
```
git add .
git commit -m ‘linting’
```

## Push your changes

Once all of these steps succeed, push your local modifications to your remote repository.

```
git push origin feature/my-contribution-branch
```

Your branch is now available on your remote forked repository, with your changes.

Next step is now to create a Pull Request so the Eurybia Team can add your changes to the official repository.

## Create a Pull Request


A pull request allows you to ask the Eurybia team to review your changes, and merge your changes into the master branch of the official repository.

To create one, on the top of your forked repository, you will find a button "Compare & pull request"

<img src="https://github.com/MAIF/eurybia/blob/master/docs/assets/images/contributing/eurybia-compare-pr.png" alt="pull request" />

As you can see, you can select on the right side which branch of your forked repository you want to associate to the pull request.

On the left side, you will find the official Eurybia repository.

- Base repository: MAIF/eurybia
- Base branch: master
- Head repository: your-github-username/eurybia
- Head branch: your-contribution-branch

<img src="https://github.com/MAIF/eurybia/blob/master/docs/assets/images/contributing/eurybia-pr-branch.png" alt="clone your forked repository" />

Once you have selected the right branch, let's create the pull request with the green button "Create pull request".

<img src="https://github.com/MAIF/eurybia/blob/master/docs/assets/images/contributing/eurybia-pr-description.png" alt="clone your forked repository" />

In the description, a template is initialized with all informations you have to give about what you are doing on what your PR is doing.

Please follow this to write your PR content.


## Finally, submit your pull request

Your pull request is now ready to be submitted. A member of the Eurybia team will contact you and will review your code and contact you if needed.

You have contributed to an Open source project, thank you and congratulations ! 🥳

Show your contribution to Eurybia in your curriculum, and share it on your social media. Be proud of yourself, you gave some code lines to the entire world !
