DOTFILES=$(wildcard *dot)
PNGFILES=$(DOTFILES:.dot=.png)
SVGFILES=$(DOTFILES:.dot=.svg)

all : $(PNGFILES) $(SVGFILES)

%.png : %.dot
	@dot -Tpng -o$@ $^

%.svg : %.dot
	@dot -Tsvg -o$@ $^
