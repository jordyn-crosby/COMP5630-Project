# Outcome specification: DIGITAL_EXCLUSION

Who Gets Left Behind? (COMP 6630, Hall and Crosby). Revision 1, November 4, 2026.
This dictionary fixes the coding rule for the binary outcome before any model is trained. It follows
the proposal (Section 3.2: "a binary indicator of digital exclusion derived from the supplement's
questions on household and personal internet use") and the completion roadmap (Section 02).

## 1. Construct

A person is digitally excluded when they do not use the Internet from any location. The CPS
Computer and Internet Use Supplement measures use, not skill, trust, accessibility, or independent
service use, so the operational outcome is Internet non-use at every location the survey asks about.
Household home access is used to audit the rule and to describe the cohort, not to define the outcome:
a person in a connected household who does not personally use the Internet anywhere is excluded; a
person in a household without home Internet who uses it elsewhere is not.

## 2. Source fields (identical names and codes in 2019, 2021 and 2023)

| Field | Level | Question (paraphrased from the supplement instrument) | Valid codes in the public file |
|---|---|---|---|
| HEINHOME | household | Does anyone in this household, including you, use the Internet at home? | 1 yes, 2 no |
| PEINHOME | person | Named as a user of the Internet at home | 1 yes, 2 no (2 for everyone when HEINHOME = 2) |
| PEINWORK | person | Named as a user at work | 1 yes, 2 no, -1 not in universe (no employed member) |
| PEINSCHL | person | Named as a user at school | 1 yes, 2 no |
| PEINCAFE | person | Named as a user at a coffee shop or other business | 1 yes, 2 no |
| PEINTRAV | person | Named as a user while traveling between places | 1 yes, 2 no |
| PEINLICO | person | Named as a user at a library, community center, park or other public place | 1 yes, 2 no |
| PEINELHO | person | Named as a user at someone else's home | 1 yes, 2 no |
| PEINOTHR | person | Named as a user at some other location | 1 yes, 2 no |
| HENETCHK | household | Follow-up in multi-person households when at least one member was not named anywhere: "it sounds like [names] do not use the Internet. Do they use the Internet from any location, for any purpose?" | 1 yes, 2 no, -1 not in universe (single-person household, or everyone already named) |
| PENETCK2 | person | Person named as a user in the HENETCHK follow-up | 1 yes, 2 not named, -1 not in universe |
| HEEVRHOM | household | Has anyone in the household ever used the Internet from home (asked when HEINHOME = 2) | 1 yes, 2 no, -1 not in universe (audit only) |

Edited-value conventions in the Census files: -1 not in universe, -2 don't know, -3 refused,
-9 no response. In the 2019, 2021 and 2023 cohort extracts none of the twelve fields above carries a
-2, -3 or -9 code, so no cohort record is unresolved; the rule still handles those codes.

Skip logic observed in the data (all three years):
* Every person with PENETCK2 = 1 also has at least one location flag = 1, because the instrument
  returns to the location questions for people named in the follow-up. The rule still treats
  PENETCK2 = 1 as use, so a file that did not back-fill the locations would be coded correctly.
* Single-person households are never asked HENETCHK (HRNUMHOU = 1 gives HENETCHK = -1); the respondent
  answered the location questions about themselves, so a "no" at every location is a direct report.
* Every apparent non-user in a multi-person household received the HENETCHK follow-up (zero cases of
  HRNUMHOU > 1, no location flag and HENETCHK = -1).
* In a household with HEINHOME = 1, members who are not home users can still be named at other
  locations; PENETCK2 = 2 appears for already-named users in follow-up households and does not mean
  "non-user" on its own.

## 3. Coding rule

```
INTERNET_USER      = 1 if (PEINHOME == 1 or PEINWORK == 1 or PEINSCHL == 1 or PEINCAFE == 1 or
                           PEINTRAV == 1 or PEINLICO == 1 or PEINELHO == 1 or PEINOTHR == 1 or
                           PENETCK2 == 1)
                     else 0
OUTCOME_STATUS     = "unresolved" if INTERNET_USER == 0 and any of the twelve fields is in {-2, -3, -9}
                     else "resolved"
DIGITAL_EXCLUSION  = 1 - INTERNET_USER  for resolved records
                   = missing            for unresolved records (counted in cohort_summary.csv, dropped
                                         from predictor_data.csv)
```

Implemented in `dataPrep.build_outcome()`. The outcome is built while every source field is still
present; the source fields are then written to `data/filtered_data.csv` (audit table) and withheld from
`data/predictor_data.csv` (modeling table). `main.py` refuses to run if any of them reaches the inputs.

## 4. Validation cases (unit tests in `test_dataPrep.TestOutcomeRule`)

| Case from the roadmap | Record | Result |
|---|---|---|
| Connected household, non-using individual | HEINHOME 1, PEINHOME 2, no other location, HENETCHK 2, PENETCK2 2 | excluded (1) |
| Individual who uses the Internet away from home | HEINHOME 2, PEINHOME 2, PEINLICO 1 | not excluded (0) |
| Household with no home Internet, no other use | HEINHOME 2, all locations 2, HENETCHK 2, PENETCK2 2 | excluded (1) |
| Follow-up corrects an apparent non-use | HEINHOME 1, PEINHOME 2, HENETCHK 1, PENETCK2 1 | not excluded (0) |
| Unresolved response | non-user with HENETCHK in {-2, -3, -9} | no outcome, status unresolved |

A home-only variable (HEINHOME = 2) would misclassify the away-from-home user, and PENETCK2 alone
would call 4,479 of the 6,610 people in the 2019 cohort "not in universe"; neither establishes the
proposed composite.

## 5. Cohort rule and counts (unweighted, six CPS categories, before outcome exclusions)

Cohort = PRTAGE >= 65 and at least one of PEDISEAR, PEDISEYE, PEDISREM, PEDISPHY, PEDISDRS, PEDISOUT
equal to 1 (team decision, November 4, 2026: all six CPS disability categories; the proposal text named
five). Overlap between categories is preserved (six 0/1 flags). The OR across the six flags reproduces
the Census any-disability flag PRDISFLG = 1 for every cohort record, which `dataPrep.audit_outcome()`
checks (`cohort_rows_without_prdisflg` must be 0).

| Year | Age 65+ | Age 65+ with any of six categories (= PRDISFLG 1) | Excluded (rule above) | Exclusion rate |
|---|---|---|---|---|
| 2019 | 22,627 | 6,610 | 2,970 | 44.9% |
| 2021 | 20,773 | 5,884 | 2,208 | 37.5% |
| 2023 | 21,478 | 6,268 | 2,030 | 32.4% |
| Total | 64,878 | 18,762 | 7,208 | 38.4% |

For reference, the five-named-category reading gives 18,226 people (6,437 / 5,725 / 6,064); the 536
people with independent-living difficulty only are now included.

`data/cohort_summary.csv` reproduces these counts (plus weighted rates, household counts, excluded
people in connected households, and users without home Internet) every time `dataPrep.py` runs.

## 6. Limits to state in the paper

CPS internet-use questions record whether a person uses the Internet, not how well, how
independently, or with what support. Exclusion as defined here therefore understates capability and
independence exclusion. Age is top-coded (80 = 80 to 84, 85 = 85 and older). Family income is a
16-bracket household variable that the Census allocates when missing (allocation flag HXFAMINC is in
the audit table). The rate of exclusion in this cohort falls from 44.9% (2019) to 32.4% (2023), so
the 2023 evaluation in Experiment B tests the models under a shifted outcome prevalence.
