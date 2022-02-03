#!/bin/bash

ls | grep "cell*" > file_name

input="file_name"
i=0
while IFS= read -r line
do
	i=$((i+1))
	echo "Text read from file: $line"
	echo "$line" > $line/readme.txt
	mv $line cell_$i 
done < $input
