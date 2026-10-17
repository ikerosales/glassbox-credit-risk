# Inteligencia Artificial Explicable

## Hands-on Project Unit 2

## Contents

1. [Formal requirements](#formal-requirements)

   * [Submission](#submission)
   * [Delivery phases and provisional
dates](#delivery-phases-and-provisional-dates)
   * [Deliverables](#deliverables)
   * [Oral defense](#oral-defense)
2. [Objective](#objective)
3. [Mandatory work](#mandatory-work)

   * [Models](#models)
   * [Evaluation](#evaluation)
   * [Interpretation](#interpretation)
4. [Report structure](#report-structure)
5. [Code](#code)
6. [Presentation structure](#presentation-structure)
7. [Grading](#grading)

## Formal requirements

### Submission

* Submit the project strictly through Moodle before each deadline.
* Make a single submission per group.
* Include the names of all group members in every relevant
deliverable.
* Do not use Moodle comments or email to provide essential submission
information.
* Submit all files for each phase in a single ZIP archive.

### Delivery phases and provisional dates

#### Phase I: preliminary submission (10%)

Submit by **25 September 2026 at 23:59**. For this phase, use only a
logical model (a decision tree, rule list, or rule set) and submit the
corresponding preliminary report and runnable notebook. This phase is
not mandatory.

#### Phase II: final submission (90%)

Submit by **20 October 2026 at 23:59**. This is the complete project and
must include all three mandatory model families, the final report and
notebook, the completed `test.csv`, and the slide deck.

### Deliverables

For both Phase I and Phase II:

1. A PDF report of at most 15 pages. Do not submit the notebook
rendered as the report.
2. A Python notebook containing the complete analysis. It must run from
top to bottom Google Colab.

For Phase II only, in addition to the above:

1. The completed `test.csv`, appending the required test predictions.
2. In Phase II, a brief PDF slide deck for the oral defense.

`test.csv` is an unlabelled test set; you will fill-in those labels
based on the predictions of a model of your choice. Evaluation partly
depends on the accuracy on this test set; failing to provide the test
set in the adequate format will be considered as having baseline
accuracy.

### Oral defense

The oral defense is mandatory and is assessed individually. Any group
member may be asked about any claim in the report, any code in the
notebook, or any submitted prediction. You may also be asked to modify
or run a small part of the code during the defense.

## Objective

Build and interpret models that predict `payment\_difficulty\_next\_month`.
The main objective is not merely to maximize predictive performance but,
also, to understand what each model has learned, explain individual
predictions, compare conflicting explanations, etc.

## Mandatory work

### Models

Fit and interpret all three of the following model families:

1. A logical model, such as a decision tree, rule list, or rule set.
2. Logistic regression.
3. A generalized additive model (Explainable Boosting Machine).

You may try other models, including black-box models, but they do not
replace the three mandatory models. You may use any learning algorithm
for the selected models.

### Evaluation

Use five-fold cross-validation on for all model comparison,
hyperparameter choices, feature selection, and classification-threshold
choices.

### Interpretation

You will need to carry out both global and local interpretation. For
local, use the same instances on all three models. Include at least one
instance from each class.

## Report structure

1. **Cover page.** Title, group members, email addresses, course, and
date.
2. **Introduction.** Problem, data, analysis strategy, and concise
overview of the main findings.
3. **Method.** Indicate your strategy and reasoning for fitting of each
of the classifiers, justifying the choice of the algorithm,
parameters and configurations. You may discuss why certain options
might make sense and while others might not.
4. **Results.** Begin with the summary table specified below, followed
by results for all models.
5. **Models.** Interpret the learned models. This is the most important
section.
6. **Discussion.** Discuss the models' performance and interpretability
(subjective). Could some aspects be improved? For example, if you
converted a tree to a rules list, could the list have been further
7. **Conclusion.** Provide conclusions regarding the project, your
approach and your results. Discuss possible additional steps towards
improvement.
8. **References.** Optional.
9. **Use of generative AI.** Briefly state who AI was used.

The Results section must begin with one row per fitted estimator class.
Adapt the last column to report the cross-validated metric or metrics
that you selected:

| Classifier | Key settings | Five-fold CV result(s) |

|---|---|---|

| `DecisionTreeClassifier` |  |  |

| `LogisticRegression` |  |  |

| `ExplainableBoostingClassifier` | `interactions=0` |  |Keep the report selective and readable. The notebook contains
implementation detail; the report should contain the evidence needed to
support the argument. Avoid raw software output, unreadable trees,
unexplained plots, and long code excerpts.

## Code

You may use standard Python packages, including scikit-learn,
InterpretML, and imodels.

## Presentation structure

You may upload presentations for the defense. Each presentation will
have four slides: one summarizing what you have done and the results
obtained, and then one for the interpretation of each of three different
types of models (e.g. tree, linear, additive).

Ideally, include some examples of local interpretation within the
slides. If you do not have space to include all the relevant work you
have done, choose what you think is the most striking or has turned out
the best.

I will not accept presentations delivered on the day of the
presentation. The presentation has to be a PDF file.

## Grading

Achieving a grade of 5 requires, at least:

* Delivering all mandatory content, as specified above;
* Not-below-chance accuracy with each model;
* Absence of major conceptual or formal mistakes.

Secondary criteria include:

* The quality and clarity of the explanations, justifications, the
writing and the presentation. Balancing clarity with rigor.
* Use of plots and other aids that help understanding the report.
* Extent of originality. Use of own data processing, methods,
analysis.

**Important:** There will be penalties for not following submission
instructions, such as exceeding the page limit, not providing the code,
etc.

Additional recommendations:

* Not everything that you do needs to appear in the report; keep the
report concise and to the point. I can check the notebook to see all
necessary details.
* Avoid plain copy-paste of code and output in the report; present the
results in a clear and concise way. In particular, do not deliver
Jupyter or RMarkdown notebooks rendered as PDF.
* Watch out with figures: for example, ensure the axis' scales are
discernible.
* I encourage discussion of the achieved results, the motivation for
trying some options and not others, and the reasoning behind the
choices made, etc.

