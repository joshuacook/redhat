#!/bin/bash
chksum=$(md5sum report.md)

while true; do
    if [ '$(chksum)' != '$(md5sum report.md)' ]; then
        echo "updating pdf"
        chksum=$(md5sum report.md)
        pandoc report.md -o report.pdf --template=template.latex
    fi
    sleep 5
done
