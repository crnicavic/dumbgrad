;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "doc"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontspec" "") ("polyglossia" "") ("geometry" "margin=2.5cm") ("amsmath" "") ("amssymb" "") ("booktabs" "") ("tabularx" "") ("array" "") ("graphicx" "") ("xcolor" "") ("tikz" "") ("listings" "") ("caption" "") ("enumitem" "") ("microtype" "") ("hyperref" "hidelinks")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "fontspec"
    "polyglossia"
    "geometry"
    "amsmath"
    "amssymb"
    "booktabs"
    "tabularx"
    "array"
    "graphicx"
    "xcolor"
    "tikz"
    "listings"
    "caption"
    "enumitem"
    "microtype"
    "hyperref")
   (TeX-add-symbols
    '("vect" 1)
    '("eng" 1)
    '("code" 1))
   (LaTeX-add-labels
    "sec:engine"
    "tab:ops"
    "eq:linear"
    "sec:graph"
    "fig:graf"
    "sec:nn"
    "eq:sos"
    "eq:ce"
    "eq:ce_softmax"
    "eq:xavier"
    "eq:adam1"
    "eq:adam2"
    "eq:adam3"
    "eq:reg"
    "eq:neuron"
    "sec:layer"
    "eq:softmax"
    "eq:lstm_f"
    "eq:lstm_i"
    "eq:lstm_g"
    "eq:lstm_o"
    "eq:lstm_c"
    "eq:lstm_h"
    "fig:odvijanje"
    "eq:bptt"
    "eq:lstm_params"
    "sec:network"
    "eq:fedsgd"
    "fig:fedsgd"
    "sec:prefiks"
    "eq:tekuci"
    "eq:prefiks"
    "eq:tacnost"
    "sec:utils"
    "eq:norm"
    "sec:primeri"
    "tab:rez")
   (LaTeX-add-bibitems
    "goodfellow"
    "lstm"
    "micrograd"
    "makemore"
    "srpeltec"
    "pypy")
   (LaTeX-add-polyglossia-langs
    '("serbian" "defaultlanguage" "script=Latin")
    '("english" "otherlanguage" "")))
 :latex)

