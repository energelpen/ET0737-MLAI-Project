"""The patient questionnaire: field definitions, BRFSS code->label maps, and parsing.

Every model feature (see the 29 names in the .meta.json) is collected here as a
human-friendly question. Coded categoricals use the *exact* BRFSS 2015 codebook
encodings that notebook 01 cleaned the training data into, so a patient's answer maps
to the same integer the model was trained on. BMI is not asked directly — we take
height + weight and compute it, which is what a clinic actually measures.
"""
from __future__ import annotations

# --------------------------------------------------------------- coded value labels
# (code -> label). These mirror notebook 01's recoding of the BRFSS 2015 codebook.
SEX = {1: "Male", 0: "Female"}

RACE = {
    1: "White (non-Hispanic)",
    2: "Black (non-Hispanic)",
    3: "American Indian / Alaska Native",
    4: "Asian",
    5: "Native Hawaiian / Pacific Islander",
    6: "Other race",
    7: "Multiracial",
    8: "Hispanic",
}

MARITAL = {
    1: "Married",
    2: "Divorced",
    3: "Widowed",
    4: "Separated",
    5: "Never married",
    6: "Member of an unmarried couple",
}

EDUCATION = {
    1: "Never attended school / only kindergarten",
    2: "Elementary (Grades 1–8)",
    3: "Some high school (Grades 9–11)",
    4: "High school graduate / GED",
    5: "Some college or technical school",
    6: "College graduate (4+ years)",
}

INCOME = {
    1: "Less than $10,000",
    2: "$10,000 – $15,000",
    3: "$15,000 – $20,000",
    4: "$20,000 – $25,000",
    5: "$25,000 – $35,000",
    6: "$35,000 – $50,000",
    7: "$50,000 – $75,000",
    8: "$75,000 or more",
}

EMPLOYMENT = {
    1: "Employed for wages",
    2: "Self-employed",
    3: "Out of work for 1 year or more",
    4: "Out of work for less than 1 year",
    5: "Homemaker",
    6: "Student",
    7: "Retired",
    8: "Unable to work",
}

GENHLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

CHECKUP = {
    1: "Within the past year",
    2: "Within the past 2 years",
    3: "Within the past 5 years",
    4: "5 or more years ago",
    5: "Never",
}

YESNO = {1: "Yes", 0: "No"}

# Readable labels for coded features, keyed by feature name — used by result/detail pages.
CODE_LABELS = {
    "Sex": SEX,
    "Race": RACE,
    "MaritalStatus": MARITAL,
    "Education": EDUCATION,
    "Income": INCOME,
    "Employment": EMPLOYMENT,
    "GenHlth": GENHLTH,
    "CheckupTime": CHECKUP,
}

# Yes/No feature -> the plain-language item shown on the review pages.
YESNO_FEATURES = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol checked in last 5 years",
    "Smoker": "Smoked 100+ cigarettes in lifetime",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Heart disease or heart attack",
    "KidneyDisease": "Kidney disease",
    "COPD": "COPD / chronic lung disease",
    "Arthritis": "Arthritis",
    "Depression": "Depressive disorder (ever told)",
    "PhysActivity": "Physically active in last 30 days",
    "Fruits": "Eats fruit at least once a day",
    "Veggies": "Eats vegetables at least once a day",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has healthcare coverage",
    "NoDocbcCost": "Skipped doctor due to cost (last year)",
    "DiffWalk": "Serious difficulty walking / climbing stairs",
}


def _yesno(key, label, help_text=None):
    return {"key": key, "label": label, "type": "yesno", "options": [(1, "Yes"), (0, "No")],
            "feature": True, "help": help_text}


def _select(key, label, options, help_text=None):
    return {"key": key, "label": label, "type": "select", "options": options,
            "feature": True, "help": help_text}


def _number(key, label, lo, hi, unit, feature=True, help_text=None):
    return {"key": key, "label": label, "type": "number", "min": lo, "max": hi,
            "unit": unit, "feature": feature, "help": help_text}


# ------------------------------------------------------------- questionnaire layout
SECTIONS = [
    {
        "id": "about", "title": "About you", "icon": "👤",
        "blurb": "Basic details that help us understand your background.",
        "fields": [
            {"key": "patient_name", "label": "Full name", "type": "text",
             "feature": False, "required": False,
             "help": "Optional — used only to identify your record at the clinic."},
            {"key": "Sex", "label": "Sex", "type": "radio",
             "options": [(1, "Male"), (0, "Female")], "feature": True, "help": None},
            _number("AgeYears", "Age", 18, 80, "years"),
            _select("Race", "Race / ethnicity", RACE),
            _select("MaritalStatus", "Marital status", MARITAL),
            _select("Education", "Highest education completed", EDUCATION),
            _select("Income", "Annual household income", INCOME),
            _select("Employment", "Employment status", EMPLOYMENT),
        ],
    },
    {
        "id": "body", "title": "Body measurements", "icon": "⚖️",
        "blurb": "We use these to calculate your Body Mass Index (BMI).",
        "fields": [
            _number("height_cm", "Height", 120, 220, "cm", feature=False),
            _number("weight_kg", "Weight", 30, 250, "kg", feature=False),
        ],
    },
    {
        "id": "conditions", "title": "Medical history", "icon": "🩺",
        "blurb": "Have you ever been told by a doctor or health professional that you have…",
        "fields": [
            _yesno("HighBP", "High blood pressure"),
            _yesno("HighChol", "High cholesterol"),
            _yesno("CholCheck", "Had your cholesterol checked in the past 5 years"),
            _yesno("HeartDiseaseorAttack", "Coronary heart disease or a heart attack"),
            _yesno("Stroke", "A stroke"),
            _yesno("KidneyDisease", "Kidney disease"),
            _yesno("COPD", "COPD, emphysema or chronic bronchitis"),
            _yesno("Arthritis", "Some form of arthritis"),
            _yesno("Depression", "A depressive disorder"),
            _yesno("DiffWalk", "Serious difficulty walking or climbing stairs"),
        ],
    },
    {
        "id": "lifestyle", "title": "Lifestyle", "icon": "🥗",
        "blurb": "Your everyday habits around diet, activity and substances.",
        "fields": [
            _yesno("Smoker", "Have you smoked at least 100 cigarettes in your life?"),
            _yesno("PhysActivity", "Any physical activity in the past 30 days (outside your job)?"),
            _yesno("Fruits", "Do you eat fruit at least once a day?"),
            _yesno("Veggies", "Do you eat vegetables at least once a day?"),
            _yesno("HvyAlcoholConsump",
                   "Heavy drinker? (men >14 drinks/week, women >7 drinks/week)"),
        ],
    },
    {
        "id": "health", "title": "Healthcare & wellbeing", "icon": "❤️",
        "blurb": "Access to care and how you've felt recently.",
        "fields": [
            _yesno("AnyHealthcare", "Do you have any kind of healthcare coverage?"),
            _yesno("NoDocbcCost",
                   "In the past year, did you need a doctor but couldn't afford one?"),
            _select("CheckupTime", "When was your last routine checkup?", CHECKUP),
            _select("GenHlth", "In general, would you say your health is…", GENHLTH),
            _number("MentHlth", "Days in the past 30 your mental health was not good", 0, 30, "days"),
            _number("PhysHlth", "Days in the past 30 your physical health was not good", 0, 30, "days"),
        ],
    },
]

# Flat lookup of every field by key.
FIELDS = {f["key"]: f for sec in SECTIONS for f in sec["fields"]}


class ValidationError(Exception):
    """Raised with a {field_key: message} dict when a submission is malformed."""

    def __init__(self, errors: dict):
        self.errors = errors
        super().__init__("questionnaire validation failed")


def parse_submission(form) -> tuple[dict, dict]:
    """Turn a posted form into (features, meta).

    `features` holds the 29 model inputs (integers/BMI float) ready for RiskModel.
    `meta` holds non-model fields we still want to store (name, height, weight).
    Raises ValidationError({field: msg}) on bad/missing input.
    """
    errors: dict = {}
    features: dict = {}
    meta: dict = {}

    def _get_int(field):
        raw = (form.get(field["key"]) or "").strip()
        if raw == "":
            errors[field["key"]] = "Please answer this question."
            return None
        try:
            val = int(raw)
        except ValueError:
            errors[field["key"]] = "Enter a whole number."
            return None
        opts = field.get("options")
        if opts is not None:
            allowed = {c for c, _ in opts} if isinstance(opts, list) else set(opts.keys())
            if val not in allowed:
                errors[field["key"]] = "Choose one of the listed options."
                return None
        if "min" in field and not (field["min"] <= val <= field["max"]):
            errors[field["key"]] = f"Enter a value between {field['min']} and {field['max']}."
            return None
        return val

    def _get_float(field):
        raw = (form.get(field["key"]) or "").strip()
        if raw == "":
            errors[field["key"]] = "Please enter a value."
            return None
        try:
            val = float(raw)
        except ValueError:
            errors[field["key"]] = "Enter a number."
            return None
        if not (field["min"] <= val <= field["max"]):
            errors[field["key"]] = f"Enter a value between {field['min']} and {field['max']}."
            return None
        return val

    for field in FIELDS.values():
        key, ftype = field["key"], field["type"]
        if ftype == "text":
            meta[key] = (form.get(key) or "").strip()
        elif ftype == "number" and not field.get("feature", True):
            val = _get_float(field)
            if val is not None:
                meta[key] = val
        elif ftype == "number":
            val = _get_int(field)
            if val is not None:
                features[key] = val
        else:  # radio / yesno / select
            val = _get_int(field)
            if val is not None:
                features[key] = val

    # Derive BMI from height + weight (kg / m^2). Clamp to the training range so a
    # typo can't push the model far outside the data it learned from.
    if "height_cm" in meta and "weight_kg" in meta:
        h_m = meta["height_cm"] / 100.0
        bmi = meta["weight_kg"] / (h_m * h_m)
        bmi = max(12.0, min(70.0, bmi))
        features["BMI"] = round(bmi, 2)

    if errors:
        raise ValidationError(errors)
    return features, meta


def humanize(feature: str, value) -> str:
    """Readable version of a stored feature value for the review pages."""
    if feature in CODE_LABELS:
        return CODE_LABELS[feature].get(int(value), str(value))
    if feature in YESNO_FEATURES:
        return YESNO.get(int(value), str(value))
    return str(value)
