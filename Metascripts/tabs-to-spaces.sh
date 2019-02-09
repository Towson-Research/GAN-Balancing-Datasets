#!/usr/bin/env bash
sed 's/\t/,/g' $1 > "${1%.*}-tabless.csv" 

