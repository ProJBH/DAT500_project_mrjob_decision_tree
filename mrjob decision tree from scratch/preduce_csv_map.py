#!/usr/bin/env python3
import sys

for line in sys.stdin:
    line = line.strip()
    value = line.split("\t")
    if len(value)>=2:
        value = ((value[1].lstrip("[")).rstrip("]")).replace(" ","")
        value = value.replace("\"","")
        print(value)

