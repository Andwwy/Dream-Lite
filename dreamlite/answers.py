"""Answer extraction and equivalence for MATH and GSM8K."""
import re
import signal
from functools import lru_cache


def last_boxed(text):
    """Contents of the last \\boxed{...} in text, or None."""
    i = text.rfind("\\boxed")
    j = text.find("{", i) if i >= 0 else -1
    if j < 0 or text[i + 6:j].strip():
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def extract_math(text):
    """Final answer of a solution: the 'final answer is' line if present, else the last box."""
    m = re.findall(r"[Tt]he final answer is(.*?)(?:\. I hope|\n|$)", text, flags=re.S)
    if m and m[-1].strip():
        return m[-1].strip().rstrip(".").strip()
    return last_boxed(text)


def normalize(a):
    if a is None:
        return None
    a = a.strip().replace("\n", "")
    if a.startswith("\\boxed"):
        a = last_boxed(a) or a
    for s in ("\\!", "\\,", "\\;", "\\ ", "\\left", "\\right", "^\\circ", "^{\\circ}", "\\$", "$",
              "\\%", "%", "\\displaystyle"):
        a = a.replace(s, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").replace("\\dots", "\\ldots")
    # units and labels in \text{...}: drop them when there is other content, else keep the words
    no_text = re.sub(r"\\(text|mbox|textbf|mathrm)\{[^}]*\}", "", a)
    a = no_text if no_text.strip(" .,") else re.sub(r"\\(text|mbox|textbf|mathrm)\{([^}]*)\}", r"\2", a)
    a = a.replace(" ", "").rstrip(".")
    if len(a.split("=")[0]) <= 2 and "=" in a:      # "x=5" -> "5"
        a = a.split("=", 1)[1]
    a = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", a)
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", a)
    a = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", a)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", a):   # 1,000 -> 1000
        a = a.replace(",", "")
    if re.fullmatch(r"-?\d+/\d+", a):                    # 1/2 -> \frac{1}{2}
        n, d = a.lstrip("-").split("/")
        a = ("-" if a.startswith("-") else "") + f"\\frac{{{n}}}{{{d}}}"
    if a.startswith("."):
        a = "0" + a
    return a or None


def _top_level_split(a):
    parts, depth, cur = [], 0, ""
    for ch in a:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


@lru_cache(maxsize=None)
def _parse(a):
    from sympy.parsing.latex import parse_latex
    try:
        return parse_latex(a)
    except _Timeout:
        raise
    except Exception:
        return None


def _sympy_equal(a, b):
    import sympy
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        x, y = _parse(a), _parse(b)
        if x is None or y is None:
            return False
        return bool(sympy.simplify(x - y) == 0)
    except (_Timeout, Exception):
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@lru_cache(maxsize=None)
def equiv(a, b):
    """Are two normalised answers the same? String match, then tuples/lists element-wise,
    then a symbolic check with a 2 s limit."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    brackets = lambda s: (s[0], s[-1]) if s[0] in "([" and s[-1] in ")]" else None
    if brackets(a) and brackets(a) == brackets(b):
        pa, pb = _top_level_split(a[1:-1]), _top_level_split(b[1:-1])
        if len(pa) == len(pb) and len(pa) > 1:
            return all(equiv(x, y) for x, y in zip(pa, pb))
    pa, pb = _top_level_split(a), _top_level_split(b)
    if len(pa) > 1 and len(pa) == len(pb) and not brackets(a):   # unordered list "1,-2"
        rest = list(pb)
        for x in pa:
            hit = next((y for y in rest if equiv(x, y)), None)
            if hit is None:
                return False
            rest.remove(hit)
        return True
    return _sympy_equal(a, b)


def vote(answers):
    """Group answers by equivalence. Returns (majority answer, indices that gave it)."""
    groups = []
    for i, a in enumerate(answers):
        if a is None:
            continue
        for g in groups:
            if equiv(g[0], a):
                g[1].append(i)
                break
        else:
            groups.append((a, [i]))
    if not groups:
        return None, []
    return max(groups, key=lambda g: len(g[1]))


def gsm_number(text):
    """Final number of a GSM8K solution as a float, or None."""
    m = re.findall(r"answer is[^\d\-]*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text)
    try:
        return float(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None
