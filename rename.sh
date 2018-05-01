#/bin/bash
find . -name '0[0-9] *' | while read -r line
do
  mv "$line" "$(echo $line | sed 's/\(0[0-9] \)/0\1/g; s/\.\///g')"
done
