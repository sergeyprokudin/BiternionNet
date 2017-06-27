# yet another simple Makefile
gcpr2017submission.pdf: gcpr2017submission.tex egbib.bib
	pdflatex gcpr2017submission.tex
	bibtex gcpr2017submission
	pdflatex gcpr2017submission.tex
	pdflatex gcpr2017submission.tex

clean:
	rm -rf *.aux *.bbl *.blg *.log
.PHONY: clean

