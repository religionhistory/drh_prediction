"""
2024-09-11 VMP

Select and preprocess answers data.
- Only group polls
- Only binary questions
- Only "Yes" and "No" answers
"""

import pandas as pd

# read data
data = pd.read_csv("../data/raw/answerset.csv")

# rename and select columns
questions = data[
    [
        "poll_name",
        "entry_id",
        "question_id",
        "question_name",
        "parent_question_id",
        "parent_question",
        "answer_value",
        "answer",
    ]
].drop_duplicates()

# recode parent questions to be 0 if nan to allow integer type
questions["parent_question_id"] = questions["parent_question_id"].fillna(0).astype(int)

""" Question Relations
Map questions between polls.
Only take questions that are in both GROUP polls.
"""

# first map questions to their related questions
question_relation = pd.read_csv("../data/raw/questionrelation.csv")
question_relation = question_relation[
    question_relation["poll_name"].str.contains("Group")
]
question_relation = question_relation[["question_id", "related_question_id"]]
questions = question_relation.merge(questions, on="question_id", how="inner")

""" Manual Fix of Problematic Case
The following questions are "related": 
"The supreme high god has knowledge of this world:" 
"The supreme high god has other knowledge of this world:" 

But this is a mistake because these are at different levels in the hierarchy,
with the latter being a sub-question of the former. We do not want the latter
question in any case because it is an "other"-type question which has [specify]
answers, so remove this now such that it does not cause problems later.
In particular by messing up the related parent question mapping. 
"""

questions = questions[
    questions["question_name"]
    != "The supreme high god has other knowledge of this world:"
]

# now map parent questions to their related parent questions
question_relation_parents = question_relation.rename(
    columns={
        "question_id": "parent_question_id",
        "related_question_id": "related_parent_question_id",
    }
)
mapping_missing_parents = pd.DataFrame(
    {"parent_question_id": [0], "related_parent_question_id": [0]}
)
question_relation_parents = pd.concat(
    [question_relation_parents, mapping_missing_parents]
)
questions = question_relation_parents.merge(
    questions, on="parent_question_id", how="inner"
)

# now remove original questions and work with related questions
questions = questions.drop(columns={"question_id", "parent_question_id"})
questions = questions.rename(
    columns={
        "related_question_id": "question_id",
        "related_parent_question_id": "parent_question_id",
    }
)

# only take the questions that appear in both polls
questions_v5 = (
    questions[questions["poll_name"] == "Religious Group (v5)"]["question_id"]
    .unique()
    .tolist()
)
questions_v6 = (
    questions[questions["poll_name"] == "Religious Group (v6)"]["question_id"]
    .unique()
    .tolist()
)
questions = questions[
    questions["question_id"].isin(questions_v5)
    & questions["question_id"].isin(questions_v6)
]

"""
Because the same Question ID can have different formulations in different
polls we now use the names from the v6 poll (could also have used v5).
We just need these to be consistent to assess duplication. 
"""

# take question names from v6 poll
questions_v6 = questions[questions["poll_name"] == "Religious Group (v6)"][
    ["question_id", "question_name"]
].drop_duplicates()

# remove columns and drop duplicates
questions = questions[
    ["entry_id", "question_id", "parent_question_id", "answer_value", "answer"]
].drop_duplicates()

# insert question names from v6 poll
questions = questions.merge(questions_v6, on="question_id", how="inner")

""" QUESTION HIERARCHY
We will need to know the placement in hierarchy of questions.
We calculate this before removing questions.
"""


# calculate level of question
# find question level for all answers
def find_question_level(question_id, drh):
    computed_levels = (
        {}
    )  # Moved inside the function to reset each call or make it an argument to preserve state across calls

    def inner_find_question_level(question_id):
        # Base case: if the parent question ID is 0, the level is 0
        if question_id == 0:
            return 0
        # If already computed, return the stored level
        if question_id in computed_levels:
            return computed_levels[question_id]

        # Recursive case: find the parent question's ID and level
        parent_id = drh.loc[
            drh["question_id"] == question_id, "parent_question_id"
        ].values[0]
        level = inner_find_question_level(parent_id) + 1
        # Store the computed level in the cache
        computed_levels[question_id] = level
        return level

    return inner_find_question_level(question_id)


question_level = questions[["question_id", "parent_question_id"]].drop_duplicates()
question_level["question_level"] = question_level["question_id"].apply(
    lambda x: find_question_level(x, question_level)
)
question_level.to_csv("../data/preprocessed/question_level.csv", index=False)


""" REMOVE NON-BINARY QUESTIONS 
We only want to use the questions that have binary (yes-no) answers.
Remove all questions that are qualitative or otherwise not binary.
"""

# get all questions and non-binary questions
questions_total = questions["question_id"].unique().tolist()
questions_binary = questions[questions["answer"].isin(["Yes", "No"])]
questions_nonbinary = list(set(questions_total) - set(questions_binary["question_id"]))

# remove questions that have [specify] in the question name
question_specify = (
    questions_binary[
        questions_binary["question_name"].str.contains(r"\[specify\]", regex=True)
    ]["question_id"]
    .unique()
    .tolist()
)  # n = 6

# remove questions that have "Other" in the question name
question_other = (
    questions_binary[questions_binary["question_name"].str.contains("Other")][
        "question_id"
    ]
    .unique()
    .tolist()
)  # n = 31

# remove additional questions not matched previously
questions_additional = [
    2892,  # "Supernatural beings care about other:",
    2878,  # "These supernatural beings possess/exhibit some other feature:",
]

# combine these into a list
questions_remove = (
    questions_nonbinary + question_specify + question_other + questions_additional
)
questions_remove = list(set(questions_remove))  # n = 57
questions = questions[~questions["question_id"].isin(questions_remove)]

""" ONLY "YES" and "NO" answers
Take only the answers that are "Yes" or "No".
Use "answer_value" because "answer" sometimes has non-English answers.
"""

questions = questions[questions["answer_value"].isin([0, 1])]
questions = questions.drop_duplicates()

""" INCONSISTENT ANSWERS
If answer is inconsistent we could do multiple things. 
We could expand, sample, or consider as missing.
We consider them missing (n = 191 cases for n = 382 total answers). 
"""

# find inconsistencies
inconsistencies = (
    questions.groupby(["entry_id", "question_id"]).size().reset_index(name="count")
)
inconsistencies = inconsistencies[inconsistencies["count"] > 1].reset_index()[
    ["entry_id", "question_id"]
]
# remove inconsistencies
questions = questions.merge(
    inconsistencies, on=["entry_id", "question_id"], how="left", indicator=True
)
questions = questions[questions["_merge"] == "left_only"].drop(columns="_merge")

# save data
questions.to_csv("../data/preprocessed/answers.csv", index=False)
