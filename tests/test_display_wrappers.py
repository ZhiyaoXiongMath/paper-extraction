from copy import deepcopy

import pytest

from paper_extraction.profile import _unwrap_display_blocks, render_tex
from paper_extraction.profile_inputs import GROUNDED_WORKFLOW, WORKFLOW_REVISION
from test_collection_edits import one_result


@pytest.mark.parametrize("env", ["conditions","relations"])
@pytest.mark.parametrize("opening,closing", [(r"\[",r"\]"),("$$","$$")])
def test_self_display_loses_only_its_redundant_outer_math_delimiter(env, opening, closing):
    inner = (r"\begin{" + env + "}\n"
             + r"&s(0_x)=x,\qquad h\circ s=h\circ\rho,\\" + "\n"
             + r"&ds_{0_x}|_{E_x}:E_x\twoheadrightarrow\ker(dh_x)." + "\n"
             + r"\end{" + env + "}")
    source = f"Suppose\n{opening}\n{inner}\n{closing}\nThen the section exists."
    assert _unwrap_display_blocks(source) == "Suppose\n" + inner + "\nThen the section exists."
    draft = one_result()
    draft["workflow_revision"] = WORKFLOW_REVISION
    draft["groups"][0]["theorem"] = source
    before = deepcopy(draft)
    assert _unwrap_display_blocks(source) in render_tex(draft)
    assert source in render_tex(draft,_original_display_wrappers=True)
    assert draft == before
    draft["workflow_revision"] = GROUNDED_WORKFLOW
    assert source in render_tex(draft)


@pytest.mark.parametrize("text", [
    r"\begin{conditions}&x>0\end{conditions}",
    r"\[\begin{aligned}x&=0\end{aligned}\]",
    r"\[\begin{cases}0&x=0\\1&x>0\end{cases}\]",
    r"\[\begin{conditions}&x=1\end{conditions}+y\]",
    r"\[x+\begin{relations}a&=b\end{relations}\]",
    r"\[\begin{conditions}&x=1\end{conditions}\begin{matrix}y\end{matrix}\]",
    r"\[\begin{relations}x&=y\end{relations}\begin{conditions}&y=1\end{conditions}\]",
    r"\[\begin{conditions}&x=1\end{relations}\]",
    r"\[\begin{conditions}&x=1\end{conditions}$$",
    r"$\text{\[\begin{conditions}&x=1\end{conditions}\]}$",
    r"\verb|\[\begin{conditions}&x=1\end{conditions}\]|",
    r"\texttt{\[\begin{conditions}&x=1\end{conditions}\]}",
    r"\\[\begin{conditions}&x=1\end{conditions}\\]",
    "% " + r"\[\begin{conditions}&x=1\end{conditions}\]",
])
def test_legal_or_ambiguous_displays_and_literal_text_keep_their_bytes(text):
    assert _unwrap_display_blocks(text) == text
