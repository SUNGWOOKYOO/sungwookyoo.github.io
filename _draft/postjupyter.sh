#!/bin/bash

# read *.ipynb* and split "*" and ".ipynb"
jupyter_file="$1"
IFS='.' read -ra INFO <<< "$1"
NAME="${INFO[0]}"

# convert "*.ipynb" file to "$NAME.md"file
jupyter nbconvert --to markdown --template jekyll.tpl "$NAME.ipynb"

# generate a posting file.
TODAY=$(date -I)
POSTNAME="$TODAY-$NAME.md"
cat template.txt > $POSTNAME
sed -i "s/DAY/$TODAY/" $POSTNAME
cat "$NAME.md" >> $POSTNAME
rm "$NAME.md"
