#!/bin/bash
TODAY=$(date -I)
POSTNAME="$TODAY-$1.md"
cat template.txt > $POSTNAME
sed -i "s/DAY/$TODAY/" $POSTNAME
