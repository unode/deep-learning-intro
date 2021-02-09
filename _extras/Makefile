DOTFILES=$(wildcard *dot)
PNGFILES=$(DOTFILES:.dot=.png)

all : $(PNGFILES)

%.png : %.dot
	@dot -Tpng -o$@ $^
