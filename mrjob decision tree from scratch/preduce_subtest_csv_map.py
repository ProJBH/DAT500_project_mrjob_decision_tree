#!/usr/bin/env python3
import sys
import re

for line in sys.stdin:
    line = re.sub("[\"\']", "", line)
    #line = line.replace('\'','')
    if line.find('Test') == 0:
            line = line.strip()
            value = line.split("\t")
            if len(value)!=0:
                value = ((value[1].lstrip("[")).rstrip("]")).replace(" ","")
                print(value)