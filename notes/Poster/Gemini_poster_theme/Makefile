.PHONY: main clean FORCE

main: poster.pdf

poster.pdf: FORCE
	latexmk -xelatex -interaction=nonstopmode -halt-on-error poster.tex

clean:
	latexmk -pdf -C
