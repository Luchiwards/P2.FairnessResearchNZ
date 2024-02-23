HTML = html
OUT = final-report.pdf
#OUT = final-report.pdf lit-survey.pdf references.pdf $(HTML)/references.html
WATCH = references.tex
HTLATEX = htlatex
MAKE4HT = make4ht
LATEXMK = latexmk -pdf

.DEFAULT_GOAL = all

FINAL_REPORT_SOURCES = final-report.tex references.bib \
	../justice-disparate-base-2001-2022.pdf \
	../police2-disparate-base.pdf

final-report.pdf : $(FINAL_REPORT_SOURCES)
references.pdf : references.tex references.bib
lit-survey.pdf : lit-survey.tex references.bib
tables-for-print.pdf : tables-for-print.tex \
	police1-disparate-impact.tex \
	police1-model-performance.tex \
	police1-model-performance-tpr.tex \
	police1-variable-importance.tex \
	police1-variable-importance-no-offence.tex

#

../justice-disparate-base-2001-2022.pdf:
	cd .. && Rscript r/justice-base-rates.R

../police2-disparate-base.pdf:
	cd .. && Rscript r/police-base-rates.R

#

.PHONY: all
all: $(OUT) | clean-tmp

.PHONY: watch
watch:
	$(LATEXMK) -pvc $(WATCH)

.PHONY: clean-tmp
clean-tmp:
	$(LATEXMK) -c
	-rm -f *.bbl *.4ct *.4tc *.idv *.lg *.tmp *.xref *.dvi

.PHONY: clean
clean: clean-tmp
	$(LATEXMK) -C
	-rm -f $(OUT)
	-rm -rf $(HTML)

#

$(HTML)/%.html : %.tex
	$(HTLATEX) $^
	if [ ! -e $(HTML) ]; then mkdir $(HTML); fi
	mv $*.html $*.css $(HTML)/

%.pdf: %.tex
	$(LATEXMK) $<
