#!/bin/bash

ls | grep "cell*" > file_name_back

input="file_name_back"
while IFS= read -r line
do
	echo "Text read from file: $line"
	value=$(<$line/readme.txt)
	mv $line "$value" 
done < $input
