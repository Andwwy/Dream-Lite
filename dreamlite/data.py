"""Datasets and prompts.

MATH train (7,500) is split into train (7,000) and dev (500) with a fixed seed.
Test sets: MATH-500, GSM8K test (1,319), and a stratified 1,000-question MMLU subset.
"""
import random
import re
from collections import defaultdict

from .answers import last_boxed

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
MATH_STOP = ["\n\nProblem:", "\nProblem:"]
GSM_STOP = ["\n\nQuestion:", "\nQuestion:"]


def math_train_dev():
    """(train, dev): 7,000 and 500 rows with keys problem, solution, type, level, answer."""
    from datasets import load_dataset
    rows = []
    for s in SUBJECTS:
        rows += list(load_dataset("EleutherAI/hendrycks_math", s, split="train"))
    for r in rows:
        r["answer"] = last_boxed(r["solution"])
    dev_idx = set(random.Random(0).sample(range(len(rows)), 500))
    return ([r for i, r in enumerate(rows) if i not in dev_idx],
            [r for i, r in enumerate(rows) if i in dev_idx])


def math_test():
    """The full MATH test set (5,000 problems), for comparison with reported scores."""
    from datasets import load_dataset
    rows = []
    for s in SUBJECTS:
        rows += list(load_dataset("EleutherAI/hendrycks_math", s, split="test"))
    for r in rows:
        r["answer"] = last_boxed(r["solution"])
    return rows


def math500():
    from datasets import load_dataset
    return list(load_dataset("HuggingFaceH4/MATH-500", split="test"))


def math_prefix(train, style="boxed"):
    """The fixed 4-shot chain-of-thought prompt.

    boxed:   four short worked solutions from MATH train, one per subject.
    minerva: the four worked examples of Lewkowycz et al. (2022), which end with
             'Final Answer: The final answer is ... I hope it is correct.'
    """
    if style == "minerva":
        return "".join(f"Problem:\n{p}\n\nSolution:\n{s}\n\n" for p, s in MINERVA_SHOTS)
    rng = random.Random(0)
    pool = [r for r in train if len(r["solution"]) < 350 and r["solution"].count("\\boxed") == 1
            and "[asy]" not in r["problem"] + r["solution"]]
    rng.shuffle(pool)
    shots, seen = [], set()
    for r in pool:
        if r["type"] not in seen:
            seen.add(r["type"])
            shots.append(r)
        if len(shots) == 4:
            break
    return "".join(f"Problem:\n{r['problem']}\n\nSolution:\n{r['solution']}\n\n" for r in shots)


def math_prompt(prefix, problem):
    return f"{prefix}Problem:\n{problem}\n\nSolution:\n"


def gsm8k(split="test"):
    from datasets import load_dataset
    rows = list(load_dataset("openai/gsm8k", "main", split=split))
    for r in rows:
        r["target"] = float(r["answer"].split("####")[-1].strip().replace(",", ""))
    return rows


def gsm_prefix():
    """Four fixed GSM8K train examples; calculator marks removed, ending 'The answer is N.'"""
    shots = random.Random(0).sample(gsm8k("train"), 4)
    out = ""
    for r in shots:
        steps, final = r["answer"].split("####")
        steps = re.sub(r"<<[^>]*>>", "", steps).strip().split("\n")
        steps = " ".join(l if l[-1] in ".!?" else l + "." for l in steps if l)
        out += f"Question: {r['question']}\nAnswer: {steps} The answer is {final.strip()}.\n\n"
    return out


def gsm_prompt(prefix, question):
    return f"{prefix}Question: {question}\nAnswer:"


def mmlu_subset(n=1000, shots=5):
    """n test questions sampled in proportion to subject size, each with its 5-shot prompt."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all")
    by_subject, dev = defaultdict(list), defaultdict(list)
    for r in ds["test"]:
        by_subject[r["subject"]].append(r)
    for r in ds["dev"]:
        dev[r["subject"]].append(r)
    total = sum(len(v) for v in by_subject.values())
    quota = {s: n * len(v) / total for s, v in by_subject.items()}
    take = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - take[s], reverse=True)[:n - sum(take.values())]:
        take[s] += 1

    def block(r, with_answer):
        opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["choices"]))
        return f"{r['question'].strip()}\n{opts}\nAnswer:" + (f" {'ABCD'[r['answer']]}\n\n" if with_answer else "")

    rng, items = random.Random(0), []
    for s in sorted(by_subject):
        head = f"The following are multiple choice questions (with answers) about {s.replace('_', ' ')}.\n\n"
        head += "".join(block(d, True) for d in dev[s][:shots])
        for r in rng.sample(by_subject[s], take[s]):
            items.append(dict(subject=s, prompt=head + block(r, False), answer=r["answer"]))
    return items


MINERVA_SHOTS = [
    ("Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
     "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, "
     "and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which "
     "gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\n"
     "Final Answer: The final answer is $[2,5)$. I hope it is correct."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = "
     "\\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how "
     "many times must Terrell lift them in order to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds "
     "of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of "
     "$2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n"
     "\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\n"
     "Final Answer: The final answer is $16$. I hope it is correct."),
    ("If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a "
     "solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we "
     "also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\n"
     "Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."),
]
